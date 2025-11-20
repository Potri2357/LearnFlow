from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db import models
from django.db.models import Avg, Count, Q

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from .models import (
    LectureNote, Question, UserAnswer, TopicWeakness,
    TopicMastery, UserStreak, StudyPlan, UserProgress
)
from .serializers import LectureNoteSerializer, QuestionSerializer, UserAnswerSerializer
from .ml_utils import extract_topics
from .utils import extract_text_from_pdf

import os
import random
import requests
import math
import datetime
import json
import re
import traceback
from datetime import timezone as dt_timezone

from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL")

client = genai.Client(api_key=GEMINI_API_KEY)

def get_current_user():
    return User.objects.first()

def get_user():
    # placeholder - swap with request.user when you add authentication
    return User.objects.first()


def call_gemini_generate(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/jso"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()



@api_view(['GET'])
def get_quiz_questions(request, note_id):
    """
    Returns up to `n` MCQs for a given lecture note.
    Query param: ?n=10  (default 10)
    """
    try:
        n = int(request.GET.get("n", 20))
    except:
        n = 20

    # Try adaptive selection first (if available)
    try:
        user = get_current_user()
        selected_questions = select_adaptive_questions(note_id, user, n=n)
        qs_data = QuestionSerializer(selected_questions, many=True).data
        return Response({"questions": qs_data})
    except Exception:
        # Fallback: return the first n saved questions (ordered by created_at)
        qs = Question.objects.filter(lecture_note_id=note_id).order_by("created_at")[:n]
        data = QuestionSerializer(qs, many=True).data
        return Response({"questions": data})


@api_view(['POST'])
def submit_mcq_answer(request):
    """
    POST: { "question_id": <id>, "selected_option": "A" }
    Updates UserAnswer and updates TopicWeakness if wrong.
    """
    user = User.objects.first()
    qid = request.data.get("question_id")
    sel = request.data.get("selected_option")

    question = Question.objects.get(id=qid)
    is_correct = (sel.upper() == question.correct_option.upper() if question.correct_option else False)

    # Save user answer (reuse UserAnswer model but now store textual answer)
    UserAnswer.objects.create(
        user=user,
        question=question,
        user_answer=sel,
        is_correct=is_correct
    )

    # Update weakness on wrong
    if not is_correct:
        topics = TopicWeakness.objects.filter(lecture_note=question.lecture_note, user=user)
        for t in topics:
            t.weakness_score += 0.2
            t.save()

    return Response({"correct": is_correct, "correct_option": question.correct_option})


# API 1: Upload Lecture Note
@api_view(['POST'])
def upload_lecture_note(request):
    user = User.objects.first()

    title = request.data.get("title")
    content = request.data.get("content")

    # ❗ Remove user=user
    note = LectureNote.objects.create(
        title=title,
        content=content
    )

    # Now extract topics and save weakness
    topics = extract_topics(content)

    for topic in topics:
        TopicWeakness.objects.create(
            user=user,
            lecture_note=note,
            topic=topic,
            weakness_score=0.0
        )

    return Response({
        "message": "Note uploaded",
        "note_id": note.id,
        "topics": topics
    })



# API 2: Generate 20 Questions From Lecture Notes without ml
@csrf_exempt
def generate_questions(request, note_id):
    # ==== 1. Get Lecture Note ====
    note = get_object_or_404(LectureNote, id=note_id)

    # ==== 2. High-quality MCQ prompt ====
    mcq_prompt = f"""
Generate exactly 5 high-quality MCQ questions from the following lecture content:

\"\"\"  
{note.content}  
\"\"\"

For each question, include:
- A clear question
- 4 options: A, B, C, and D
- The correct option letter only (A/B/C/D)
- An explanation for the correct answer
- A difficulty score between 0.2 (easy) and 0.9 (hard)

Return ONLY JSON in this format:

[
  {{
    "question": "What is ...?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": "B",
    "explanation": "Because ...",
    "difficulty": 0.6
  }}
]
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": mcq_prompt}
                ]
            }
        ]
    }

    # ==== 3. Send to Gemini ====
    try:
        response = requests.post(
            GEMINI_API_URL + f"?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload
        )

        raw = response.json()
        print("RAW GEMINI QUESTION OUTPUT:", raw)

        # Extract text
        output_text = raw["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return JsonResponse({"error": f"Gemini error: {e}"}, status=500)

    # ==== 4. Safe JSON parsing (handles bad output) ====
    try:
        # Fix common formatting errors
        cleaned = output_text.strip()

        # Sometimes Gemini adds backticks or Markdown
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        questions_json = json.loads(cleaned)

    except Exception:
        return JsonResponse({
            "error": "Gemini JSON parsing failed",
            "raw_output": output_text
        }, status=400)

    # ==== 5. Create Question objects ====
    saved_questions = []

    for q in questions_json:
        try:
            question = Question.objects.create(
                lecture_note=note,
                question_text=q["question"],
                option_a=q["options"][0],
                option_b=q["options"][1],
                option_c=q["options"][2],
                option_d=q["options"][3],
                correct_option=q["correct"].strip().upper(),
                explanation=q.get("explanation", ""),
                difficulty=float(q.get("difficulty", 0.5)),
            )
            saved_questions.append(QuestionSerializer(question).data)

        except Exception as e:
            print("ERROR saving question:", e)
            continue  # skip bad entries

    return JsonResponse({
        "generated_count": len(saved_questions),
        "questions": saved_questions
    })


# API 3: Submit Answer
@api_view(['POST'])
def submit_answer(request):
    user = User.objects.first()
    question_id = request.data.get("question_id")
    user_answer = request.data.get("user_answer")

    question = Question.objects.get(id=question_id)
    is_correct = (user_answer.lower() == question.correct_answer.lower())

    # Save answer
    UserAnswer.objects.create(
        user=user,
        question=question,
        user_answer=user_answer,
        is_correct=is_correct
    )

    # Update weakness score (if wrong)
    if not is_correct:
        # find related topics of this lecture note
        print("UPDATING WEAKNESS...")
        topics = TopicWeakness.objects.filter(
            lecture_note=question.lecture_note,
            user=user
        )

        for t in topics:
            t.weakness_score += 0.2
            t.save()
            
    print("IS CORRECT:", is_correct)
    print("QUESTION ID:", question.id)
    print("QUESTION NOTE ID:", question.lecture_note.id)


    return Response({"correct": is_correct})


# API 4: Get Weak Topics
@api_view(['GET'])
def weak_topics(request):
    user = User.objects.first()
    note_id = request.GET.get("note_id")

    if not note_id:
        return Response({"error": "note_id is required"}, status=400)

    try:
        note = LectureNote.objects.get(id=note_id)
    except LectureNote.DoesNotExist:
        return Response({"error": "Invalid note_id"}, status=404)

    weaknesses = TopicWeakness.objects.filter(
        user=user,
        lecture_note=note
    ).order_by('-weakness_score')

    data = [
        {"topic": w.topic, "score": w.weakness_score}
        for w in weaknesses
    ]

    return Response({"weak_topics": data})




# API 5: Get Progress
@api_view(['GET'])
def progress(request):
    user = User.objects.first()

    answers = UserAnswer.objects.filter(user=user)
    total = answers.count()
    correct = answers.filter(is_correct=True).count()

    accuracy = (correct / total) * 100 if total > 0 else 0

    return Response({
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy
    })
    

def get_user():
    # temporary: return first user (replace with request.user when auth implemented)
    return User.objects.first()

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def update_topic_mastery(user, lecture_note, topic, delta):
    """
    Update or create TopicMastery for (user, lecture_note, topic).
    delta is additive (can be positive or negative).
    We'll clamp mastery to [0,1].
    Return new mastery.
    """
    tm, created = TopicMastery.objects.get_or_create(
        user=user, lecture_note=lecture_note, topic=topic,
        defaults={"mastery": 0.3}
    )
    tm.mastery = clamp01(tm.mastery + delta)
    tm.save()
    return tm.mastery

def set_user_streak(user, topic, correct):
    """
    Maintain per-topic and global streaks.
    If correct: increment; else reset to 0.
    """
    # global streak (topic=None)
    global_streak, _ = UserStreak.objects.get_or_create(user=user, topic=None, defaults={"streak":0})
    # topic-based streak
    topic_streak, _ = UserStreak.objects.get_or_create(user=user, topic=topic, defaults={"streak":0})

    if correct:
        global_streak.streak += 1
        topic_streak.streak += 1
    else:
        global_streak.streak = 0
        topic_streak.streak = 0

    global_streak.save()
    topic_streak.save()
    return global_streak.streak, topic_streak.streak

def compute_user_accuracy(user):
    answers = UserAnswer.objects.filter(user=user)
    total = answers.count()
    if total == 0:
        return 0.5
    correct = answers.filter(is_correct=True).count()
    return correct / total

def select_adaptive_questions(note_id, user, n=10):
    """
    Selection algorithm:
    1) Find topic masteries for this note; sort ascending (weakest first).
    2) Take half (n//2) questions from weak topics (prefer lower difficulty).
    3) Take rest from mix chosen to match target difficulty (based on user accuracy).
    4) Avoid questions that user recently answered (last 50 answers).
    Returns queryset/list of Question objects in desired order.
    """
    note = LectureNote.objects.get(id=note_id)
    # fetch all questions for this note
    all_questions = list(Question.objects.filter(lecture_note=note))
    if not all_questions:
        return []

    # recent answered question ids
    recent_q_ids = list(UserAnswer.objects.filter(user=user).order_by('-answered_at').values_list('question_id', flat=True)[:100])

    # compute mastery per topic; if missing, fallback to TopicWeakness table topics (or extracted topics)
    tm_qs = TopicMastery.objects.filter(user=user, lecture_note=note).order_by('mastery')
    if tm_qs.exists():
        weak_topics = [t.topic for t in tm_qs[:max(1, len(tm_qs))]]
    else:
        # fallback: use TopicWeakness objects (reverse sorted)
        tw = TopicWeakness.objects.filter(user=user, lecture_note=note).order_by('-weakness_score')
        weak_topics = [t.topic for t in tw][:5]

    # map topic -> list of questions that mention the topic (simple substring match)
    topic_to_qs = {}
    for q in all_questions:
        text = (q.question_text or "").lower()
        for t in weak_topics:
            if t.lower() in text:
                topic_to_qs.setdefault(t, []).append(q)

    # compute target difficulty from accuracy
    acc = compute_user_accuracy(user)
    # map acc to target difficulty (0..1)
    # simple linear map: acc 0.0 -> diff 0.35 (easier), acc 1.0 -> diff 0.85 (harder)
    target_diff = 0.35 + acc * (0.85 - 0.35)

    selected = []
    used_ids = set(recent_q_ids)  # avoid recent questions

    # 50% from weak topics
    weak_quota = n // 2
    for topic in weak_topics:
        candidates = [q for q in (topic_to_qs.get(topic, [])) if q.id not in used_ids]
        # sort by difficulty ascending (easier first to rebuild base)
        candidates.sort(key=lambda x: getattr(x, "difficulty", 0.5))
        while candidates and len(selected) < weak_quota:
            selected.append(candidates.pop(0))
            used_ids.add(selected[-1].id)
        if len(selected) >= weak_quota:
            break

    # Fill rest based on target difficulty
    rest_quota = n - len(selected)
    # candidate pool: questions not used and not in recent
    pool = [q for q in all_questions if q.id not in used_ids]
    # sort pool by closeness to target_diff
    pool.sort(key=lambda q: abs(getattr(q, "difficulty", 0.5) - target_diff))
    for q in pool:
        if len(selected) >= n:
            break
        selected.append(q)
        used_ids.add(q.id)

    # if still less than n, fill with any not recent
    if len(selected) < n:
        for q in all_questions:
            if q.id in used_ids: continue
            selected.append(q)
            used_ids.add(q.id)
            if len(selected) >= n: break

    # final: cut to n
    return selected[:n]

# --- new endpoint: adaptive quiz start ---

@api_view(['POST'])
def adaptive_quiz_start(request):
    """
    Request body: { "note_id": <int>, "n": 10 }
    Returns: JSON list of serialized questions (MCQ fields).
    """
    user = get_user()
    note_id = request.data.get("note_id")
    n = int(request.data.get("n", 10))
    try:
        selected = select_adaptive_questions(note_id, user, n=n)
    except LectureNote.DoesNotExist:
        return Response({"error": "Invalid note_id"}, status=400)

    qs_data = QuestionSerializer(selected, many=True).data
    return Response({"questions": qs_data})

# --- override/extend submit_mcq_answer to update mastery + streaks ---

@api_view(['POST'])
def submit_mcq_answer(request):
    """
    POST { "question_id": <id>, "selected_option": "A" }
    Updates UserAnswer + TopicMastery + TopicWeakness + UserProgress
    """
    try:
        user = get_user()
        qid = request.data.get("question_id")
        sel = request.data.get("selected_option","").strip().upper()
        if not qid:
            return Response({"error":"question_id required"}, status=400)

        question = get_object_or_404(Question, id=qid)
        is_correct = (sel == (question.correct_option or "").upper())

        ua = UserAnswer.objects.create(
            user=user,
            question=question,
            user_answer=sel,
            is_correct=is_correct
        )

        # Primary topic is question.topic
        primary_topic = question.topic or "general"

        # update topic mastery
        tm, created = TopicMastery.objects.get_or_create(user=user, lecture_note=question.lecture_note, topic=primary_topic, defaults={"mastery":0.30})
        # learning rate based on difficulty
        qdiff = float(question.difficulty or 0.5)
        lr = 0.08
        if qdiff > 0.7:
            lr = 0.12
        elif qdiff < 0.35:
            lr = 0.05

        if is_correct:
            delta = lr * (1.0 - tm.mastery)
        else:
            delta = - (lr * 0.6) * tm.mastery

        tm.mastery = max(0.0, min(1.0, tm.mastery + delta))
        tm.save()

        # Update TopicWeakness (increase when wrong)
        tw, _ = TopicWeakness.objects.get_or_create(user=user, lecture_note=question.lecture_note, topic=primary_topic, defaults={"weakness_score": 0.0})
        if not is_correct:
            tw.weakness_score = round(tw.weakness_score + 0.2, 3)
            tw.save()
        else:
            # small decay in weakness if correct
            tw.weakness_score = max(0.0, round(tw.weakness_score - 0.08, 3))
            tw.save()

        #optional: Update aggregate user progress (if you have a model)
        up, _ = UserProgress.objects.get_or_create(user=user)
        up.total_questions += 1
        if is_correct:
             up.correct_answers += 1
        up.save()

        return Response({
            "correct": is_correct,
            "correct_option": question.correct_option,
            "updated_mastery": { primary_topic: tm.mastery },
            "weakness_score": tw.weakness_score
        })

    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


def parse_numbered_sections(text):
    """
    Parse a plan that has numbered sections:
      1. Strength Topics:
      2. Weak Topics to Focus On:
      3. Recommended Learning Resources:
         Articles:
         Videos:
         Explanations:
      4. Practice Plan:
      5. Revision Plan:
      6. Next Assessment:
    Return dict of named sections (strings)
    """
    # normalize newlines
    text = text.replace("\r\n", "\n")

    # main section headings (only these trigger a section change)
    main_headings = [
        ("strengths", ["strength topics", "strengths", "strength topic", "strength"]),
        ("weak", ["weak topics", "weak topics to focus on", "weak", "weaknesses", "weak topics to focus"]),
        ("resources", ["recommended learning resources", "recommended resources", "resources"]),
        ("practice", ["practice plan", "practice", "practice plan:"]),
        ("revision", ["revision plan", "revision"]),
        ("assessment", ["next assessment", "next assessment:", "assessment"]),
    ]

    # Prepare mapping
    mapping = {k: [] for k, _ in main_headings}

    current = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            # preserve blank lines as separator
            if current:
                mapping[current].append("")
            continue

        low = line.lower()

        # detect MAIN heading lines (with or without leading number)
        found = False
        for key, tokens in main_headings:
            for tok in tokens:
                # match patterns like '1. Strength Topics:' or 'Strength Topics:' or 'Strength Topics'
                if low.startswith(tok) or re.match(r"^\d+\.\s*" + re.escape(tok), low):
                    # capture any text after the ':' on the same line
                    # e.g., '3. Recommended Learning Resources: Articles:' -> keep 'Articles:' as next content
                    parts = re.split(r":\s*", raw_line, maxsplit=1)
                    if len(parts) > 1 and parts[1].strip():
                        mapping[key].append(parts[1].strip())
                    current = key
                    found = True
                    break
            if found:
                break

        if found:
            continue

        # not a main heading line: append to current section if any
        # (this includes subheadings like "Articles:", "Videos:", "Easy:", "Medium:" which stay under current section)
        if current:
            mapping[current].append(raw_line.rstrip())

    # join lines into text blocks
    result = {k: "\n".join([l for l in v]).strip() for k, v in mapping.items()}

    return result

@api_view(['POST'])
def generate_study_plan(request):
    """
    Build study plan using current mastery + weak topics and call Gemini (same client).
    """
    try:
        note_id = request.data.get("note_id")
        note = get_object_or_404(LectureNote, id=note_id)
        user = get_user()

        mastery_qs = TopicMastery.objects.filter(user=user, lecture_note=note)
        strengths = {m.topic: round(m.mastery, 2) for m in mastery_qs if m.mastery >= 0.65}
        weak_topics = {m.topic: round(m.mastery, 2) for m in mastery_qs if m.mastery < 0.45}

        # Safe fallback (so LLM always has something)
        if not strengths:
            strengths = {"General understanding": 0.30}
        if not weak_topics:
            weak_topics = {"Key concepts to practice": 0.25}

        strengths_text = "\n".join([f"- {k}: mastery {v}" for k,v in strengths.items()])
        weaknesses_text = "\n".join([f"- {k}: mastery {v}" for k,v in weak_topics.items()])

        prompt = f"""
You are an AI tutor. Generate a structured study plan for the student based on the data below.

Lecture note: {note.title}

Strength Topics:
{strengths_text}

Weak Topics:
{weaknesses_text}

Requirements:
- Return EXACTLY this STRUCTURE below, nothing else.
1. Strength Topics:
- <topic>: why student is strong (1-2 sentences)

2. Weak Topics to Focus On:
- <topic>: short explanation why weak (1-2 sentences)

3. Recommended Learning Resources:
Articles:
- <article1>
- <article2>
- <article3> 
(atleast 3 points)
Videos:
- <video1>
- <video2>
(atleast 3 points)
Explanations:
- <explain1>
- <explain2>
(atleast 3 points)

4. Practice Plan:
Easy:
- <task1>
- <task2>
- <task3>
Medium:
- <task1>
- <task2>
- <task3>
Hard:
- <task1>
- <task2>
- <task3>

5. Revision Plan:
- <item1>
- <item2>

6. Next Assessment:
- <recommendation1>
- <recommendation2>

Rules:
- No markdown, no code block, no JSON
- Provide at least 2 items per list
- Be concise and actionable
"""

        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        # get text robustly
        try:
            plan_text = result.text
        except Exception:
            try:
                plan_text = result.candidates[0].content.parts[0].text
            except:
                plan_text = str(result)

        # clean obvious wrappers
        plan_text = plan_text.replace("```", "").strip()

        # parse into structured sections for frontend
        try:
            plan_sections = parse_numbered_sections(plan_text)
        except Exception:
            plan_sections = {
                "strengths": "",
                "weak": "",
                "resources": "",
                "practice": "",
                "revision": "",
                "assessment": "",
            }

        # Return both new and backward-compatible keys
        return Response({
            "plan": plan_text,
            "plan_sections": plan_sections,
            "sections": plan_sections,  # legacy alias
            "strengths": strengths,
            "weak_topics": weak_topics,
        })

    except Exception as e:
        traceback.print_exc()
        return Response({"error": "Study plan generation failed", "details": str(e)}, status=500)






@api_view(["GET"])
def analytics_for_note(request, note_id):
    user = get_current_user()

    try:
        note = LectureNote.objects.get(id=note_id)
    except LectureNote.DoesNotExist:
        return Response({"error": "invalid note_id"}, status=400)

    # -------------------------------------------------------
    # 1) TOPIC MASTERY (improved)
    # -------------------------------------------------------
    tm_qs = TopicMastery.objects.filter(user=user, lecture_note=note)

    topic_mastery = []
    for t in tm_qs:
        topic_mastery.append({
            "topic": t.topic,
            "mastery": round(t.mastery, 3),
            "last_updated": t.last_updated,
        })

    return Response({
        "topic_mastery": topic_mastery
    })

@api_view(['GET'])
def get_note_details(request, note_id):
    """
    Returns details about a lecture note, including the total number of available questions.
    """
    try:
        note = LectureNote.objects.get(id=note_id)
        question_count = Question.objects.filter(lecture_note=note).count()
        return Response({
            "id": note.id,
            "title": note.title,
            "question_count": question_count
        })
    except LectureNote.DoesNotExist:
        return Response({"error": "Note not found"}, status=404)

    # Weighted mastery score
    mastery_score = 0.0
    if topic_mastery:
        mastery_score = (
            sum([t["mastery"] * 1.2 for t in topic_mastery]) /
            len(topic_mastery)
        ) * 100

    # -------------------------------------------------------
    # 2) WEAK TOPICS (improved accuracy)
    # -------------------------------------------------------
    tw_qs = TopicWeakness.objects.filter(user=user, lecture_note=note)

    # Only consider topics with real activity
    active_weak = []
    for tw in tw_qs:
        # Normalize weakness
        score = min(max(tw.weakness_score, 0), 5)
        active_weak.append({
            "topic": tw.topic,
            "weakness_score": round(score, 2),
        })

    # Sort highest weakness first
    active_weak.sort(key=lambda x: x["weakness_score"], reverse=True)

    top_weak = active_weak[:5]

    # -------------------------------------------------------
    # 3) DIFFICULTY ACCURACY (major improvement)
    # -------------------------------------------------------
    answers = UserAnswer.objects.filter(user=user, question__lecture_note=note)

    def bucket(low, high):
        qset = answers.filter(
            question__difficulty__gte=low,
            question__difficulty__lt=high
        )
        total = qset.count()
        correct = qset.filter(is_correct=True).count()
        accuracy = (correct * 100 / total) if total else None
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 2) if accuracy is not None else None,
        }

    difficulty_accuracy = {
        "easy": bucket(0.0, 0.4),
        "medium": bucket(0.4, 0.7),
        "hard": bucket(0.7, 1.1),
    }

    # -------------------------------------------------------
    # 4) ACCURACY TREND (actual 7-day performance)
    # -------------------------------------------------------
    today = timezone.now().date()
    trend = []

    for i in range(6, -1, -1):
        day = today - datetime.timedelta(days=i)

        start = datetime.datetime.combine(
        day, datetime.time.min, tzinfo=dt_timezone.utc
        )
        end = datetime.datetime.combine(
        day, datetime.time.max, tzinfo=dt_timezone.utc
        )

        qs = answers.filter(answered_at__range=(start, end))
        total = qs.count()
        correct = qs.filter(is_correct=True).count()

        acc = (correct * 100 / total) if total else None

        trend.append({
            "date": day.isoformat(),
            "accuracy": round(acc, 2) if acc is not None else None,
            "total": total,
        })

    # -------------------------------------------------------
    # 5) RECENT SESSIONS
    # -------------------------------------------------------
    recent = list(
        answers.order_by("-answered_at")[:20].values(
            "answered_at",
            "is_correct",
            "question__question_text",
            "question__difficulty"
        )
    )

    recent_sessions = [
        {
            "ts": r["answered_at"],
            "question": (
                r["question__question_text"][:120] + "..."
                if len(r["question__question_text"]) > 120
                else r["question__question_text"]
            ),
            "difficulty": float(r["question__difficulty"]),
            "is_correct": r["is_correct"],
        }
        for r in recent
    ]

    return Response({
        "mastery_score": round(mastery_score, 2),
        "topic_mastery": topic_mastery,
        "top_weak_topics": top_weak,
        "difficulty_accuracy": difficulty_accuracy,
        "accuracy_trend_last7": trend,
        "recent_sessions": recent_sessions,
    })



@api_view(["GET"])
def recent_weak_topics(request, note_id=None):
    """
    GET /api/recent-weak-topics/?note_id=#
    Returns top 5 weak topics for user/note
    """
    user = get_current_user()
    note = None
    if note_id:
        try:
            note = LectureNote.objects.get(id=note_id)
        except LectureNote.DoesNotExist:
            note = None

    qs = TopicWeakness.objects.filter(user=user)
    if note:
        qs = qs.filter(lecture_note=note)
    qs = qs.order_by("-weakness_score")[:10]
    result = [{"topic": t.topic, "weakness_score": round(t.weakness_score,3)} for t in qs]
    return Response({"top_weak_topics": result})


@api_view(["POST"])
def next_actions(request):
    """
    POST /api/next-actions/  (body: { note_id: <int> })
    Returns simple heuristic recommendations (local).
    Optionally you can call Gemini here — commented code shows where.
    """
    user = get_current_user()
    note_id = request.data.get("note_id")
    try:
        note = LectureNote.objects.get(id=note_id)
    except:
        return Response({"error": "invalid note_id"}, status=400)

    # pick 3 weakest topics
    tw = TopicWeakness.objects.filter(user=user, lecture_note=note).order_by("-weakness_score")[:5]
    if not tw.exists():
        # fallback to topic mastery lowest
        tm = TopicMastery.objects.filter(user=user, lecture_note=note).order_by("mastery")[:5]
        candidates = [{"topic": t.topic, "mastery": t.mastery} for t in tm]
    else:
        candidates = [{"topic": t.topic, "weakness_score": t.weakness_score} for t in tw]

    # simple heuristic recommendations
    recs = []
    for c in candidates[:3]:
        topic = c.get("topic")
        recs.append({
            "topic": topic,
            "recommendation": f"Review short notes on '{topic}', practice 5 easy questions, then try 3 medium questions."
        })

    return Response({"recommendations": recs})

@api_view(["GET"])
def ai_insights(request, note_id):

    # -----------------------------
    # Build Analytics from DB
    # -----------------------------
    try:
        progress = UserProgress.objects.filter(lecture_note_id=note_id)

        mastery_score = (
            sum([p.mastery for p in progress]) / len(progress)
            if progress.exists() else 0
        )

        accuracy_trend = [
            {"date": p.date.strftime("%Y-%m-%d"), "accuracy": p.accuracy}
            for p in progress.order_by("date")[:7]
        ]

        weak_topics = list(
            TopicWeakness.objects.filter(note_id=note_id)
            .values("topic", "weakness_score")
        )

        difficulty_accuracy = {
            "easy": {"accuracy": progress.filter(difficulty="easy").aggregate(avg=models.Avg("accuracy"))["avg"] or 0},
            "medium": {"accuracy": progress.filter(difficulty="medium").aggregate(avg=models.Avg("accuracy"))["avg"] or 0},
            "hard": {"accuracy": progress.filter(difficulty="hard").aggregate(avg=models.Avg("accuracy"))["avg"] or 0},
        }

        analytics = {
            "mastery_score": mastery_score,
            "accuracy_trend_last7": accuracy_trend,
            "top_weak_topics": weak_topics,
            "difficulty_accuracy": difficulty_accuracy,
        }

    except Exception as e:
        print("ANALYTICS ERROR:", e)
        return Response({"insights": f"ANALYTICS ERROR: {e}"})

    # -----------------------------
    # Build Prompt
    # -----------------------------
    prompt = f"""
Generate clean learning insights based ONLY on the following analytics data:

{json.dumps(analytics, indent=2)}

Rules:
- No disclaimers
- No introductions
- No “I need more data”
- No repeating the data
- Direct insights only: strengths, weak areas, recommendations
"""

    # -----------------------------
    # Call Gemini API
    # -----------------------------
    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        "gemini-2.5-flash:generateContent?key=" + GEMINI_API_KEY
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(url, json=payload)
        print("RAW GEMINI OUTPUT:", response.text)

        data = response.json()

        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
        ai_text = ai_text.replace("**", "").strip()

        return Response({"insights": ai_text})

    except Exception as e:
        print("GEMINI ERROR:", e)
        return Response({"insights": ""})

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_pdf(request):
    """
    Upload PDF, extract text, save LectureNote, seed topics in TopicWeakness and TopicMastery
    """
    try:
        user = get_user()
        title = request.data.get("title", request.FILES.get("file").name if request.FILES.get("file") else "Untitled")
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"error": "No PDF uploaded"}, status=400)

        extracted_text = extract_text_from_pdf(file_obj)

        note = LectureNote.objects.create(
            user=user,
            title=title,
            file=file_obj,
            content=extracted_text
        )

        # extract topics and seed weakness + mastery
        topics = extract_topics(extracted_text)

        if not topics:
            topics = ["general"]

        for topic in topics:
            TopicWeakness.objects.create(user=user, lecture_note=note, topic=topic, weakness_score=0.0)
            TopicMastery.objects.get_or_create(user=user, lecture_note=note, topic=topic, defaults={"mastery": 0.30})

        serializer = LectureNoteSerializer(note)
        return Response({"note_id": note.id, "topics": topics, "message": "PDF uploaded + topics saved", "note": serializer.data}, status=201)

    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)




def extract_json_array(text: str):
    """
    Bracket-counting extractor: returns the first complete JSON array string in text,
    or None if no complete array found.
    """
    if not text:
        return None
    start = text.find('[')
    if start == -1:
        return None
    bracket_count = 0
    end_index = None
    for i, ch in enumerate(text[start:], start):
        if ch == '[':
            bracket_count += 1
        elif ch == ']':
            bracket_count -= 1
        if bracket_count == 0:
            end_index = i
            break
    if end_index is None:
        return None
    return text[start:end_index+1]

@api_view(['POST'])
def generate_mcqs(request):
    """
    Generate MCQs via Gemini and persist questions including topic field.
    Request body: { "note_id": <int>, "count": 10 }
    """
    try:
        note_id = request.data.get("note_id")
        count = int(request.data.get("count", 10))
        note = get_object_or_404(LectureNote, id=note_id)
        content = note.content or ""

        prompt = f"""
Generate exactly {count} high-quality multiple-choice questions (MCQs) covering the full content below.
For each question, return a JSON object with these keys:
- topic: short topic name (single phrase)
- question: the question text
- options: an array of 4 strings (A,B,C,D)
- answer: the correct letter (A/B/C/D)
- explanation: a 1-2 sentence explanation
- difficulty: "easy" or "medium" or "hard"

Return a JSON array ONLY (no markdown, no text before/after).
Content:
\"\"\"{content}\"\"\"
"""

        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        # Some client responses may be in .text or a nested structure; handle both
        raw_text = None
        try:
            raw_text = result.text
        except Exception:
            # fallback to nested candidate content (safe)
            try:
                raw_text = result.candidates[0].content.parts[0].text
            except Exception:
                raw_text = str(result)

        cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()

        # sometimes LLM prints extra text; attempt to find first JSON array
        import re
        m = re.search(r'\[.*\]', cleaned, flags=re.DOTALL)
        json_str = m.group(0) if m else cleaned

        mcqs = json.loads(json_str)

        saved = []
        for item in mcqs:
            topic = item.get("topic") or (item.get("question").split()[0][:30] if item.get("question") else "general")
            options = item.get("options", [])
            # ensure 4 options
            while len(options) < 4:
                options.append("")
            q = Question.objects.create(
                lecture_note=note,
                topic=topic,
                question_text=item.get("question",""),
                option_a=options[0],
                option_b=options[1],
                option_c=options[2],
                option_d=options[3],
                correct_option=item.get("answer","").strip().upper()[:1],
                explanation=item.get("explanation",""),
                difficulty=0.5 if item.get("difficulty") is None else (0.3 if item.get("difficulty")=="easy" else (0.6 if item.get("difficulty")=="medium" else 0.85))
            )
            saved.append(QuestionSerializer(q).data)
            # Ensure TopicMastery exists for this topic
            TopicMastery.objects.get_or_create(user=note.user or get_user(), lecture_note=note, topic=topic, defaults={"mastery": 0.30})

        return Response({"generated_count": len(saved), "questions": saved})

    except Exception as e:
        traceback.print_exc()
        return Response({"error":"Gemini request failed", "details": str(e)}, status=500)











