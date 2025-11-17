from django.shortcuts import render

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import LectureNote, Question, UserAnswer, TopicWeakness, UserProgress
from .serializers import LectureNoteSerializer, QuestionSerializer, UserAnswerSerializer
from django.contrib.auth.models import User
from .ml_utils import extract_topics
from .models import TopicWeakness
import random
import os
import requests
import random
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth.models import User
from .models import LectureNote, Question, TopicWeakness, UserAnswer
from .serializers import QuestionSerializer
from .ml_utils import extract_topics  # if you still use it
import math
from django.utils import timezone
from rest_framework import status
from .models import TopicMastery, UserStreak , StudyPlan
from django.db.models import Avg, Count, Q
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
import datetime
import json
from django.http import JsonResponse
from django.conf import settings
import re 
from core.models import LectureNote, UserProgress, TopicWeakness
import requests
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL")

def get_current_user():
    return User.objects.first()


def call_gemini_generate(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
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


@api_view(["POST"])
def generate_mcqs(request):
    note_id = request.data.get("note_id")
    count = int(request.data.get("count", 10))

    note = LectureNote.objects.get(id=note_id)

    prompt = f"""
Generate {count} multiple-choice questions (MCQs) from the following text.
Return ONLY a JSON array in this format:

[
  {{
    "question": "text",
    "options": ["A", "B", "C", "D"],
    "correct": 0,
    "explanation": "short explanation",
    "difficulty": 0.4
  }}
]

Content:
\"\"\"{note.content}\"\"\"
"""

    try:
        result = call_gemini_generate(prompt)
        raw = result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return Response({"error": "Gemini request failed", "details": str(e)}, status=500)

    # Extract JSON array using regex
    import json, re
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return Response({"error": "Could not parse JSON", "raw": raw}, status=500)

    try:
        mcqs = json.loads(match.group(0))
    except:
        return Response({"error": "Invalid JSON format", "raw": raw}, status=500)

    # Save MCQs to database
    saved = []
    for item in mcqs:
        opts = item.get("options", [])
        while len(opts) < 4:
            opts.append("")

        correct_letter = ["A", "B", "C", "D"][ item.get("correct", 0) ]

        q = Question.objects.create(
            lecture_note=note,
            question_text=item.get("question", ""),
            option_a=opts[0],
            option_b=opts[1],
            option_c=opts[2],
            option_d=opts[3],
            correct_option=correct_letter,
            explanation=item.get("explanation", ""),
            difficulty=float(item.get("difficulty", 0.5))
        )
        saved.append(QuestionSerializer(q).data)

    return Response({"questions": saved})


@api_view(['GET'])
def get_quiz_questions(request, note_id):
    """
    Returns all MCQs for a given lecture note.
    """
    qs = Question.objects.filter(lecture_note_id=note_id)
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

    note = LectureNote.objects.create(
        user=user,
        title=title,
        content=content
    )

    # extract topics automatically
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

    latest_note = LectureNote.objects.filter(user=user).last()
    weaknesses = TopicWeakness.objects.filter(
        user=user,
        lecture_note=latest_note
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
    Updates DB and topic mastery adaptively.
    """
    user = get_user()
    qid = request.data.get("question_id")
    sel = request.data.get("selected_option")
    try:
        question = Question.objects.get(id=qid)
    except Question.DoesNotExist:
        return Response({"error": "Invalid question id"}, status=400)

    is_correct = False
    if question.correct_option:
        is_correct = (sel.upper() == question.correct_option.upper())

    # Save user answer
    ua = UserAnswer.objects.create(
        user=user,
        question=question,
        user_answer=sel,
        is_correct=is_correct
    )

    # Determine related topics for this question:
    # We use the TopicMastery rows for this lecture note; if none, fall back to TopicWeakness topics.
    topics = list(TopicMastery.objects.filter(user=user, lecture_note=question.lecture_note).values_list('topic', flat=True))
    if not topics:
        # fallback
        topics = list(TopicWeakness.objects.filter(user=user, lecture_note=question.lecture_note).values_list('topic', flat=True))

    # If still empty: try extracting topics from question text (use ml_utils.extract_topics)
    if not topics:
        try:
            topics = extract_topics(question.question_text)
        except Exception:
            topics = []

    # update mastery per topic (distribute delta across topics)
    if topics:
        # base delta magnitude depending on difficulty
        # easier questions less effect, harder questions more effect
        qdiff = float(getattr(question, "difficulty", 0.5))
        # learning rate factor
        lr = 0.08  # base learning rate
        if qdiff > 0.7:
            lr = 0.12
        elif qdiff < 0.35:
            lr = 0.05

        # adjust sign
        if is_correct:
            # increase mastery, scaled by how much room to improve
            for t in topics:
                # compute delta as lr * (1 - current_mastery)
                curr = TopicMastery.objects.filter(user=user, lecture_note=question.lecture_note, topic=t).first()
                curr_mastery = curr.mastery if curr else 0.3
                delta = lr * (1.0 - curr_mastery)
                update_topic_mastery(user, question.lecture_note, t, delta)
        else:
            # wrong answer reduces mastery modestly
            for t in topics:
                curr = TopicMastery.objects.filter(user=user, lecture_note=question.lecture_note, topic=t).first()
                curr_mastery = curr.mastery if curr else 0.3
                delta = - (lr * 0.6) * curr_mastery  # reduce proportional to current mastery
                update_topic_mastery(user, question.lecture_note, t, delta)

    # update streaks
    # pick primary topic (first in topics) if exists, else None
    primary_topic = topics[0] if topics else None
    global_streak, topic_streak = set_user_streak(user, primary_topic, is_correct)

    # Also keep TopicWeakness in sync: increment weakness score for wrong
    if not is_correct:
        tws = TopicWeakness.objects.filter(lecture_note=question.lecture_note, user=user)
        for tw in tws:
            tw.weakness_score += 0.2
            tw.save()

    # Update UserProgress model (optional)
    up, _ = UserProgress.objects.get_or_create(user=user)
    up.total_questions += 1
    if is_correct:
        up.correct_answers += 1
    up.save()

    return Response({
        "correct": is_correct,
        "correct_option": question.correct_option,
        "current_mastery": { t: TopicMastery.objects.filter(user=user, lecture_note=question.lecture_note, topic=t).first().mastery for t in topics } if topics else {}
    })

@api_view(['POST'])
def generate_study_plan(request):
    user = User.objects.first()  # Replace with request.user when auth added
    note_id = request.data.get("note_id")

    try:
        note = LectureNote.objects.get(id=note_id)
    except LectureNote.DoesNotExist:
        return Response({"error": "Invalid note_id"}, status=400)

    # 1. Topic Mastery
    mastery = list(TopicMastery.objects.filter(user=user, lecture_note=note)
                   .values("topic", "mastery"))

    # 2. Weaknesses
    weaknesses = list(TopicWeakness.objects.filter(user=user, lecture_note=note)
                      .values("topic", "weakness_score"))

    # 3. Recent mistakes
    recent_answers = UserAnswer.objects.filter(user=user).order_by('-answered_at')[:30]
    mistakes = []
    for a in recent_answers:
        if not a.is_correct:
            mistakes.append(a.question.question_text)

    # 4. Prepare summary
    summary = {
        "lecture_title": note.title,
        "mastery": mastery,
        "weaknesses": weaknesses,
        "mistakes": mistakes,
    }

    # 5. Gemini Request
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }



    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            {
                "role": "user",
                "content": f"""
You are an AI tutor. Create a detailed personalized study plan based on this data:

{summary}

Your output must follow this EXACT structure:

STUDY PLAN
-----------
1. Weak Topics to Focus On:
   - <topic>: Explanation + why student is weak

2. Recommended Learning Resources:
   - Videos, articles, short explanations

3. Practice Plan:
Easy:
- (at least 3 tasks)
Medium:
- (at least 3 tasks)
Hard:
- (at least 3 tasks)

Make sure each difficulty level has at least 3 bullet points.
NEVER leave them empty.

4. Revision Plan:
   - Summary tasks, quick notes, important points

5. Next Assessment:
   - Adaptive quiz recommendation

Return the plan as pure text. Do NOT add JSON.
"""
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.3
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

# Try OpenAI-style
    plan_text = None
    try:
        plan_text = data["choices"][0]["message"]["content"]
    except:
        pass

# Try Gemini-native
    if plan_text is None:
        try:
            plan_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            pass

# Try Gemini list root
    if plan_text is None:
        try:
            plan_text = data[0]["candidates"][0]["content"]["parts"][0]["text"]
        except:
            pass

    if plan_text is None:
        return Response({"error": "Unable to parse Gemini response", "raw": data}, status=500)


    # Save plan
    StudyPlan.objects.create(user=user, lecture_note=note, plan_text=plan_text)
    return Response({"plan": plan_text})

@api_view(["GET"])
def analytics_for_note(request, note_id):
    """
    GET /api/analytics/<note_id>/
    Returns:
      - mastery_score (0-100)
      - topic_mastery: [{topic, mastery}]
      - top_weak_topics: [{topic, mastery, weakness_score}]
      - difficulty_accuracy: {easy: {correct, total, acc}, medium: {...}, hard: {...}}
      - accuracy_trend_last7: [{date, accuracy}]  -> overall accuracy per day (last 7 days)
      - recent_sessions: [{ts, question, is_correct, difficulty}]
      - study_plans: [{created_at, snippet}]
    """
    user = get_current_user()
    try:
        note = LectureNote.objects.get(id=note_id)
    except LectureNote.DoesNotExist:
        return Response({"error": "invalid note_id"}, status=400)

    # 1) Topic mastery list
    tm_qs = TopicMastery.objects.filter(user=user, lecture_note=note)
    topic_mastery = [
        {"topic": t.topic, "mastery": round(t.mastery, 3), "last_updated": t.last_updated}
        for t in tm_qs
    ]
    mastery_score = 0.0
    if topic_mastery:
        mastery_score = sum([t["mastery"] for t in topic_mastery]) / len(topic_mastery) * 100

    # 2) top weak topics (use TopicWeakness if available, else by mastery ascending)
    tw_qs = TopicWeakness.objects.filter(user=user, lecture_note=note).order_by("-weakness_score")[:8]
    if tw_qs.exists():
        top_weak = [{"topic": t.topic, "weakness_score": round(t.weakness_score, 3)} for t in tw_qs[:5]]
    else:
        sorted_by_mastery = sorted(topic_mastery, key=lambda x: x["mastery"])[:5]
        top_weak = [{"topic": t["topic"], "mastery": t["mastery"]} for t in sorted_by_mastery]

    # 3) Difficulty accuracy buckets based on question difficulty
    answers = UserAnswer.objects.filter(user=user, question__lecture_note=note)
    def bucket_stats(qs, low, high):
        subset = qs.filter(question__difficulty__gte=low, question__difficulty__lt=high)
        total = subset.count()
        correct = subset.filter(is_correct=True).count()
        acc = (correct / total) * 100 if total else None
        return {"total": total, "correct": correct, "accuracy": round(acc,2) if acc is not None else None}

    diff_easy = bucket_stats(answers, 0.0, 0.4)
    diff_med  = bucket_stats(answers, 0.4, 0.7)
    diff_hard = bucket_stats(answers, 0.7, 1.1)

    # 4) accuracy trend last 7 days (overall)
    today = timezone.now().date()
    trend = []
    for i in range(6, -1, -1):  # 6 days ago ... today
        d = today - datetime.timedelta(days=i)
        day_start = datetime.datetime.combine(d, datetime.time.min).replace(tzinfo=datetime.timezone.utc)
        day_end = datetime.datetime.combine(d, datetime.time.max).replace(tzinfo=datetime.timezone.utc)
        day_qs = answers.filter(answered_at__range=(day_start, day_end))
        total = day_qs.count()
        correct = day_qs.filter(is_correct=True).count()
        acc = (correct / total)*100 if total else None
        trend.append({"date": d.isoformat(), "accuracy": round(acc,2) if acc is not None else None, "total": total})

    # 5) recent sessions (last 20 answers)
    recent = list(answers.order_by("-answered_at")[:20].values("answered_at", "is_correct", "question__question_text", "question__difficulty"))
    recent_sessions = [
        {
            "ts": r["answered_at"],
            "question": (r["question__question_text"][:120] + ("..." if len(r["question__question_text"])>120 else "")) if r["question__question_text"] else "",
            "is_correct": r["is_correct"],
            "difficulty": round(r["question__difficulty"] or 0.5, 2)
        } for r in recent
    ]

    # 6) recent study plans
    plans = StudyPlan.objects.filter(user=user, lecture_note=note).order_by("-created_at")[:6]
    study_plans = [{"created_at": p.created_at, "snippet": p.plan_text[:300] + ("..." if len(p.plan_text)>300 else "")} for p in plans]

    payload = {
        "mastery_score": round(mastery_score,2),
        "topic_mastery": topic_mastery,
        "top_weak_topics": top_weak,
        "difficulty_accuracy": {"easy": diff_easy, "medium": diff_med, "hard": diff_hard},
        "accuracy_trend_last7": trend,
        "recent_sessions": recent_sessions,
        "study_plans": study_plans
    }

    return Response(payload)


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
    import requests
    import json
    from django.conf import settings
    from django.db import models
    from core.models import UserProgress, TopicWeakness

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
        "gemini-2.5-flash:generateContent?key=" + settings.GEMINI_API_KEY
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















