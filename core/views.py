from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db import models
from django.db.models import Avg, Count, Q

from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status, permissions

from .models import (
    LectureNote, Question, UserAnswer, TopicWeakness,
    TopicMastery, UserStreak, StudyPlan, UserProgress, Notification
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

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL")

genai.configure(api_key=GEMINI_API_KEY)

def get_current_user(request=None):
    """Get the current authenticated user or fallback to first user"""
    if request and hasattr(request, 'user') and request.user.is_authenticated:
        return request.user
    # Fallback for views that don't pass request
    return User.objects.first()

def get_user(request=None):
    """Alias for get_current_user for backward compatibility"""
    return get_current_user(request)


def clean_option_text(text):
    """
    Remove letter prefixes (A), B), C), D)) from option text.
    This ensures options are stored without prefixes in the database.
    The frontend will add the prefixes when displaying.
    """
    if not text:
        return ""
    # Remove patterns like "A) ", "B) ", "C) ", "D) " from the start
    import re
    cleaned = re.sub(r'^[A-D]\)\s*', '', str(text).strip(), flags=re.IGNORECASE)
    return cleaned.strip()


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
    user = request.user if request.user.is_authenticated else User.objects.first()
    qid = request.data.get("question_id")
    sel = request.data.get("selected_option")
    time_taken = int(request.data.get("time_taken", 0))

    question = Question.objects.get(id=qid)
    is_correct = (sel.upper() == question.correct_option.upper() if question.correct_option else False)

    # Save user answer
    UserAnswer.objects.create(
        user=user,
        question=question,
        user_answer=sel,
        is_correct=is_correct,
        time_taken=time_taken
    )

    # Update weakness based on correctness AND time
    # If question has a specific topic, try to update that topic's weakness
    # Otherwise fallback to updating all topics for the note (legacy behavior)
    
    topics_to_update = []
    if question.topic:
        qs = TopicWeakness.objects.filter(lecture_note=question.lecture_note, user=user, topic__iexact=question.topic)
        if qs.exists():
            topics_to_update = list(qs)
    
    if not topics_to_update:
        # Fallback to all topics if specific topic not found
        topics_to_update = list(TopicWeakness.objects.filter(lecture_note=question.lecture_note, user=user))

    # ML-inspired weakness scoring with time analysis
    import math
    
    for t in topics_to_update:
        # Get question difficulty (0.2-0.9, default 0.5)
        difficulty = question.difficulty if question.difficulty else 0.5
        
        if not is_correct:
            # Wrong answer: increase weakness proportional to difficulty
            # Harder questions contribute more to weakness
            weakness_increase = 0.15 + (difficulty * 0.15)  # Range: 0.18-0.285
            t.weakness_score += weakness_increase
        else:
            # Correct answer: analyze time performance
            # Use exponential decay: faster = better mastery
            # Expected time based on difficulty: harder = more time allowed
            expected_time = 15 + (difficulty * 15)  # Range: 18-28.5 seconds
            time_ratio = time_taken / expected_time
            
            if time_ratio > 1.5:  # Much slower than expected
                # Still struggling despite correct answer
                t.weakness_score += 0.08 * difficulty
            elif time_ratio > 1.0:  # Slightly slower
                # Minor weakness indicator
                t.weakness_score += 0.03 * difficulty
            elif time_ratio < 0.5:  # Very fast (strong mastery)
                # Exponential mastery bonus for very fast answers
                mastery_gain = 0.15 * (1 - time_ratio) * (1 + difficulty)
                t.weakness_score = max(0.0, t.weakness_score - mastery_gain)
            elif time_ratio < 0.8:  # Fast (good mastery)
                # Linear mastery gain for reasonably fast answers
                mastery_gain = 0.1 * (1 - time_ratio) * (1 + difficulty * 0.5)
                t.weakness_score = max(0.0, t.weakness_score - mastery_gain)
            # else: time_ratio between 0.8-1.0 = neutral (no change)
        
        # Apply bounds: weakness score should stay in reasonable range
        t.weakness_score = max(0.0, min(2.0, t.weakness_score))
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
                option_a=clean_option_text(q["options"][0]),
                option_b=clean_option_text(q["options"][1]),
                option_c=clean_option_text(q["options"][2]),
                option_d=clean_option_text(q["options"][3]),
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

    # FINAL FALLBACK: If we still don't have enough, reuse recent questions (ignoring used_ids check for recent ones)
    if len(selected) < n:
        # Get all questions that were skipped because they were in recent_q_ids
        # We need to check against the CURRENT used_ids (which includes selected ones)
        # to avoid duplicates in the current selection.
        remaining_needed = n - len(selected)
        
        # Candidates are questions that are NOT in the current selection
        # (i.e. they are in all_questions but NOT in selected)
        # We can just iterate all_questions again and pick ones not in selected.
        
        # Create a set of currently selected IDs for fast lookup
        selected_ids_set = {q.id for q in selected}
        
        for q in all_questions:
            if q.id not in selected_ids_set:
                selected.append(q)
                selected_ids_set.add(q.id) # Mark as selected
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


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def quiz_completed(request):
    """
    POST: { "note_id": <id>, "score": <int>, "total": <int> }
    Triggers notification creation for quiz completion
    """
    try:
        print(f"DEBUG: quiz_completed called. User: {request.user}, Auth: {request.auth}")
        print(f"DEBUG: Data: {request.data}")
        
        user = request.user  # Use request.user directly since we require authentication
        note_id = request.data.get("note_id")
        score = int(request.data.get("score", 0))
        total = int(request.data.get("total", 1))
        
        if not note_id:
            print("DEBUG: note_id missing")
            return Response({"error": "note_id required"}, status=400)
        
        note = get_object_or_404(LectureNote, id=note_id)
        current_score = (score / total * 100) if total > 0 else 0
        
        print(f"DEBUG: Creating notification for {user.username}, Score: {current_score}%")
        
        # Import here to avoid circular import
        from .signals import create_quiz_completion_notification
        create_quiz_completion_notification(user, note, current_score, total)
        
        print("DEBUG: Notification created successfully")
        return Response({"message": "Notification created", "score": current_score})
    except Exception as e:
        print(f"DEBUG: Error in quiz_completed: {e}")
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
    Build study plan using current mastery + weak topics and call Gemini.
    Tries 'gemini-2.0-flash-exp' first, falls back to 'gemini-1.5-flash'.
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

        def generate_with_model(model_name):
            print(f"Attempting study plan generation with {model_name}...")
            model = genai.GenerativeModel(model_name)
            result = model.generate_content(prompt)
            return result

        # Try primary model (fast/lite), then fallback (latest stable)
        try:
            result = generate_with_model('gemini-2.0-flash-lite')
        except Exception as e:
            print(f"Primary model (gemini-2.0-flash-lite) failed: {e}")
            print("Falling back to gemini-flash-latest...")
            try:
                result = generate_with_model('gemini-flash-latest')
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                raise e2

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
            "date": day.strftime("%m-%d"),
            "accuracy": round(acc, 1) if acc is not None else 0
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


        # Use the call_gemini_generate function which has the correct API format
        response_data = call_gemini_generate(prompt)
        
        # Extract text from response
        raw_text = None
        if isinstance(response_data, dict):
            try:
                raw_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                try:
                    # Fallback: try to get text directly
                    raw_text = response_data.get("text", str(response_data))
                except Exception:
                    raw_text = str(response_data)
        else:
            raw_text = str(response_data)

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
                option_a=clean_option_text(options[0]),
                option_b=clean_option_text(options[1]),
                option_c=clean_option_text(options[2]),
                option_d=clean_option_text(options[3]),
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






# ============================================
# AUTHENTICATION VIEWS
# ============================================

from rest_framework import generics, permissions, status
from rest_framework.views import APIView
from .serializers import RegisterSerializer, UserProfileSerializer, NotificationSerializer, UserSerializer

class RegisterView(generics.CreateAPIView):
    """
    POST /api/auth/register/
    Register a new user
    """
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]


class UserProfileView(generics.RetrieveUpdateAPIView):
    """
    GET/PUT /api/profile/
    Get or update user profile
    """
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user.profile


class NotificationListView(generics.ListAPIView):
    """
    GET /api/notifications/
    List all notifications for the authenticated user
    """
    serializer_class = NotificationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Notification.objects.filter(user=self.request.user).order_by('-created_at')


class NotificationMarkReadView(APIView):
    """
    POST /api/notifications/<id>/mark-read/
    Mark a notification as read
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        try:
            notification = Notification.objects.get(pk=pk, user=request.user)
            notification.is_read = True
            notification.save()
            return Response({'status': 'notification marked as read'})
        except Notification.DoesNotExist:
            return Response({'error': 'Notification not found'}, status=status.HTTP_404_NOT_FOUND)


class NotificationMarkAllReadView(APIView):
    """
    POST /api/notifications/mark-all-read/
    Mark all notifications as read for the user
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        return Response({'status': 'all notifications marked as read'})


class NotificationDeleteView(APIView):
    """
    DELETE /api/notifications/<id>/delete/
    Delete a notification
    """
    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, pk):
        try:
            notification = Notification.objects.get(pk=pk, user=request.user)
            notification.delete()
            return Response({'status': 'notification deleted'})
        except Notification.DoesNotExist:
            return Response({'error': 'Notification not found'}, status=status.HTTP_404_NOT_FOUND)


class CurrentUserView(APIView):
    """
    GET /api/auth/me/
    Get current authenticated user info
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)


@api_view(['GET', 'PUT'])
@permission_classes([permissions.IsAuthenticated])
def user_profile(request):
    """
    GET /api/profile/
    Get user profile with stats
    
    PUT /api/profile/
    Update user bio
    """
    user = request.user
    
    if request.method == 'GET':
        # Calculate stats
        total_quizzes = UserAnswer.objects.filter(user=user).values('question__lecture_note').distinct().count()
        
        # Calculate average score
        user_answers = UserAnswer.objects.filter(user=user)
        if user_answers.exists():
            correct_count = user_answers.filter(is_correct=True).count()
            total_count = user_answers.count()
            average_score = round((correct_count / total_count) * 100) if total_count > 0 else 0
        else:
            average_score = 0
        
        # Get streak days
        try:
            streak = UserStreak.objects.filter(user=user).first()
            streak_days = streak.current_streak if streak else 0
        except:
            streak_days = 0
        
        # Get or create user profile data (using User model for now)
        profile_data = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'bio': getattr(user, 'bio', ''),  # Will add this field if needed
            'total_quizzes': total_quizzes,
            'average_score': average_score,
            'streak_days': streak_days,
            'date_joined': user.date_joined.isoformat() if user.date_joined else None,
        }
        
        return Response(profile_data)
    
    elif request.method == 'PUT':
        # Update bio (for now, we'll store it in a simple way)
        # In production, you'd want a UserProfile model
        bio = request.data.get('bio', '')
        # For now, just return success
        # TODO: Add UserProfile model to store bio
        return Response({'status': 'success', 'message': 'Profile updated'})


