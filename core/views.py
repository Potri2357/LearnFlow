from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db import models
from django.db.models import Avg, Count, Q, Sum as ModelsSum

from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status, permissions

from .models import (
    LectureNote, Question, UserAnswer, TopicWeakness,
    TopicMastery, UserStreak, StudyPlan, UserProgress, Notification, QuizAttempt,
    ExamSyllabus, ExamConfiguration, UserProfile, StickyNote
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
import time
from datetime import timezone as dt_timezone

import google.generativeai as genai
from .ai_utils import generate_ai_content

# Import exam preparation views
from .exam_views import (
    upload_exam_syllabus, upload_previous_papers, generate_exam_questions,
    get_exam_questions, update_exam_question, delete_exam_question, list_exam_syllabi,
    generate_exam_strategy
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL")
# genai.configure(api_key=GEMINI_API_KEY) # Comment out if not used, or leave as is if other parts use it directly

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
    return generate_ai_content(prompt)



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
        
        # --- SYNC TO TOPIC MAINTAIN ---
        # Mastery = 1.0 - (Weakness / 2.0)
        # 0.0 weakness => 1.0 mastery (100%)
        # 1.0 weakness => 0.5 mastery (50%)
        # 2.0 weakness => 0.0 mastery (0%)
        
        try:
            mst_val = max(0.0, 1.0 - (t.weakness_score / 2.0))
            TopicMastery.objects.update_or_create(
                user=user,
                lecture_note=question.lecture_note,
                topic=t.topic,
                defaults={"mastery": mst_val}
            )
        except Exception as e:
            print(f"Error syncing mastery: {e}")

    return Response({"correct": is_correct, "correct_option": question.correct_option})


# API 1: Upload Lecture Note
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def upload_lecture_note(request):
    try:
        user = request.user

        title = request.data.get("title")
        content = request.data.get("content")

        if not title or not content:
            return Response({"error": "Title and content are required."}, status=400)

        # Check for duplicate content
        existing_note = LectureNote.objects.filter(user=user, content=content).first()
        if existing_note:
            return Response({
                "message": "Note already exists.",
                "note_id": existing_note.id,
                "topics": []
            }, status=200)

        note = LectureNote.objects.create(
            user=user,
            title=title,
            content=content
        )

        # Now extract topics and save weakness
        try:
            topics = extract_topics(content)
        except Exception as e:
            print(f"Topic extraction failed: {e}")
            topics = ["general"]

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
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)



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

    # ==== 3. Send to AI (Strategy: Try Llama -> Fallback to Gemini) ====
    output_text = ""
    source = "Llama"
    
    try:
        # 1. Try Llama (User Preference)
        try:
            output_text = generate_ai_content(mcq_prompt, model="meta-llama/llama-3.3-70b-instruct:free")
            if not output_text or not output_text.strip():
                raise Exception("Empty response from Llama")
            
            # Quick check if it looks like JSON
            if "{" not in output_text and "[" not in output_text:
                 raise Exception("Response does not look like JSON")
                 
            print("RAW Llama OUTPUT:", output_text)
            
        except Exception as e:
            print(f"Llama Generation failed ({e}). Falling back to Gemini...")
            source = "Gemini"
            # 2. Fallback to Gemini
            gemini_response = generate_with_gemini(mcq_prompt)
            output_text = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
            print("RAW Gemini OUTPUT:", output_text)

    except Exception as e:
        return JsonResponse({"error": f"All AI Generation failed: {e}"}, status=500)

    # ==== 4. Safe JSON parsing (handles bad output) ====
    try:
        # Fix common formatting errors
        cleaned = output_text.strip()
        
        # Remove markdown code blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
             cleaned = cleaned.split("```")[1].split("```")[0]
             
        cleaned = cleaned.strip()
        questions_json = json.loads(cleaned)

    except Exception:
        return JsonResponse({
            "error": "JSON parsing failed",
            "raw_output": output_text,
            "source": source
        }, status=400)
        
        # Remove markdown code blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
             cleaned = cleaned.split("```")[1].split("```")[0]
             
        cleaned = cleaned.strip()
        questions_json = json.loads(cleaned)

    except Exception:
        return JsonResponse({
            "error": "JSON parsing failed",
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

    # Create notification for question generation
    if saved_questions and note.user:
        Notification.objects.create(
            user=note.user,
            message=f"✅ {len(saved_questions)} questions generated successfully for '{note.title}'! Ready to practice."
        )

    return JsonResponse({
        "generated_count": len(saved_questions),
        "questions": saved_questions
    })


# API 3: Submit Answer
@api_view(['POST'])
def submit_answer(request):
    try:
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
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


# API 4: Get Weak Topics
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def weak_topics(request):
    try:
        user = request.user
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
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)




# API 5: Get Progress
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def progress(request):
    try:
        user = request.user

        answers = UserAnswer.objects.filter(user=user)
        total = answers.count()
        correct = answers.filter(is_correct=True).count()

        accuracy = (correct / total) * 100 if total > 0 else 0

        return Response({
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": accuracy
        })
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
    

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
@permission_classes([permissions.IsAuthenticated])
def adaptive_quiz_start(request):
    """
    Request body: { "note_id": <int>, "n": 10 }
    Returns: JSON list of serialized questions (MCQ fields).
    """
    user = request.user
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
@permission_classes([permissions.IsAuthenticated])
def submit_mcq_answer(request):
    """
    POST { "question_id": <id>, "selected_option": "A" }
    Updates UserAnswer + TopicMastery + TopicWeakness + UserProgress
    """
    try:
        user = request.user
        qid = request.data.get("question_id")
        sel = request.data.get("selected_option","").strip().upper()
        time_taken = int(request.data.get("time_taken", 0))

        if not qid:
            return Response({"error":"question_id required"}, status=400)

        question = get_object_or_404(Question, id=qid)
        is_correct = (sel == (question.correct_option or "").upper())

        ua = UserAnswer.objects.create(
            user=user,
            question=question,
            user_answer=sel,
            is_correct=is_correct,
            time_taken=time_taken
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
        
        # Save QuizAttempt
        from .models import QuizAttempt
        QuizAttempt.objects.create(
            user=user,
            lecture_note=note,
            score=score,
            total_questions=total
        )
        
        # Import here to avoid circular import
        from .signals import create_quiz_completion_notification
        create_quiz_completion_notification(user, note, current_score, total)
        
        # --- Update Streak ---
        from django.utils import timezone
        import datetime
        
        # Global Streak (topic=None)
        user_streak, created = UserStreak.objects.get_or_create(user=user, topic=None)
        
        now = timezone.now()
        today = now.date()
        last_date = user_streak.last_updated.date() if not created else (today - datetime.timedelta(days=1))
        
        if created:
            user_streak.streak = 1
            user_streak.save()
        else:
            # If validated today, do nothing (streak already counted or maintained)
            # Unless we want to count distinct sessions? usually streak is daily.
            if last_date < today:
                if last_date == (today - datetime.timedelta(days=1)):
                    # Consecutive day
                    user_streak.streak += 1
                else:
                    # Broken streak
                    user_streak.streak = 1
                user_streak.save()
        
        return Response({"message": "Notification created", "score": current_score, "streak": user_streak.streak})
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
    Accepts optional parameters: exam_date, hours_per_day, priority_subjects, focus_weak_areas.
    """
    try:
        note_id = request.data.get("note_id")
        exam_date = request.data.get("exam_date")
        hours_per_day = request.data.get("hours_per_day")
        priority_subjects = request.data.get("priority_subjects", [])
        focus_weak_areas = request.data.get("focus_weak_areas", False)

        note = get_object_or_404(LectureNote, id=note_id)
        user = request.user

        mastery_qs = TopicMastery.objects.filter(user=user, lecture_note=note)
        strengths = {m.topic: round(m.mastery, 2) for m in mastery_qs if m.mastery >= 0.65}
        
        # Get weak topics
        weakness_qs = TopicWeakness.objects.filter(user=user, lecture_note=note).order_by("-weakness_score")
        weak_topics = {w.topic: round(w.weakness_score, 2) for w in weakness_qs if w.weakness_score > 0.3}

        # Safe fallback (so LLM always has something)
        if not strengths:
            strengths = {"General understanding": 0.30}
        if not weak_topics:
            weak_topics = {"Key concepts to practice": 0.25}

        strengths_text = "\n".join([f"- {k}: mastery {v}" for k,v in strengths.items()])
        weaknesses_text = "\n".join([f"- {k}: mastery {v}" for k,v in weak_topics.items()])

        # Build prompt context
        user_context = ""
        if exam_date:
            user_context += f"Target Exam Date: {exam_date}\n"
        if hours_per_day:
            user_context += f"Daily Study Commitment: {hours_per_day} hours\n"
        if priority_subjects:
            user_context += f"Priority Subjects (User Selected): {', '.join(priority_subjects) if isinstance(priority_subjects, list) else priority_subjects}\n"
        if focus_weak_areas:
            user_context += "EMPHASIS: Focus heavily on weak areas.\n"

        prompt = f"""
You are an AI tutor. Generate a structured study plan for the student based on the data below.

Lecture note: {note.title}

Student Profile:
{user_context}

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

Rules:
- No markdown, no code block, no JSON
- Provide at least 2 items per list
- Be concise and actionable
"""

        # Try primary model (verified available)
        try:
            # Returns plain text - use Mistral for speed
            plan_text = generate_ai_content(prompt, model="mistralai/mistral-7b-instruct:free")
        except Exception as e:
            print(f"AI generation failed: {e}")
            raise e

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
@permission_classes([permissions.IsAuthenticated])
def analytics_for_note(request, note_id):
    user = request.user

    try:
        # Try to get note for this user
        note = LectureNote.objects.get(id=note_id, user=user)
    except LectureNote.DoesNotExist:
        # Fallback: check if note exists but has no user (legacy data)
        try:
            note = LectureNote.objects.get(id=note_id, user__isnull=True)
            # Assign the note to current user
            note.user = user
            note.save()
        except LectureNote.DoesNotExist:
            return Response({"error": "invalid note_id or permission denied"}, status=404)

    # -------------------------------------------------------
    # 1) MASTERY SCORE - Based on actual quiz performance
    # -------------------------------------------------------
    # Get all answers for this note
    all_answers = UserAnswer.objects.filter(user=user, question__lecture_note=note)
    total_answers = all_answers.count()
    correct_answers = all_answers.filter(is_correct=True).count()
    
    # Calculate mastery score from actual performance
    if total_answers > 0:
        mastery_score = (correct_answers / total_answers) * 100
    else:
        mastery_score = 0.0
    
    # Topic Mastery (for display)
    tm_qs = TopicMastery.objects.filter(user=user, lecture_note=note)
    topic_mastery = []
    for t in tm_qs:
        topic_mastery.append({
            "topic": t.topic,
            "mastery": round(t.mastery, 3),
            "last_updated": t.last_updated,
        })

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
    # Call AI API
    # -----------------------------
    try:
        # Use fast Mistral for insights
        data = generate_ai_content(prompt, model="mistralai/mistral-7b-instruct:free")

        # data is now simple text string
        ai_text = data.replace("**", "").strip()

        return Response({"insights": ai_text})

    except Exception as e:
        print("GEMINI ERROR:", e)
        return Response({"insights": "Analysis unavailable due to service load."})

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([permissions.IsAuthenticated])
def upload_pdf(request):
    """
    Upload PDF, extract text, save LectureNote, seed topics in TopicWeakness and TopicMastery
    """
    from .ocr_utils import process_document_pipeline
    
    try:
        user = request.user
        title = request.data.get("title", request.FILES.get("file").name if request.FILES.get("file") else "Untitled")
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"error": "No PDF uploaded"}, status=400)

        # 1. Attempt Digital Extraction first
        initial_text = extract_text_from_pdf(file_obj)
        
        # 2. Run Advanced Pipeline (OCR check -> Cleanup -> Compression)
        # Reset file pointer for OCR reading if needed
        file_obj.seek(0)
        final_content = process_document_pipeline(file_obj=file_obj, extracted_text_if_digital=initial_text)

        # 3. Validation: If content is still empty/too short, fail gracefully
        if not final_content or len(final_content.strip()) < 20:
             return Response({
                 "error": "Could not extract text from this document. If this is a scanned PDF, OCR tools (Tesseract/Poppler) may be missing on the server. Please upload a digital PDF.",
                 "details": "OCR Extraction Failed"
             }, status=422)

        # Check for duplicate content (using final processed content)
        existing_note = LectureNote.objects.filter(user=user, content=final_content).first()
        if existing_note:
             serializer = LectureNoteSerializer(existing_note)
             return Response({
                 "note_id": existing_note.id, 
                 "topics": [], 
                 "message": "Note with same content already exists.", 
                 "note": serializer.data
             }, status=200)

        note = LectureNote.objects.create(
            user=user,
            title=title,
            file=file_obj,
            content=final_content # Storing the CLEANED, COMPRESSED text
        )

        # extract topics and seed weakness + mastery
        topics = extract_topics(final_content)

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

        if not content.strip():
            return Response({"error": "This lecture note has no text content. Upload a PDF with readable text first."}, status=400)

        prompt = f"""You are an expert educational AI. Generate exactly {count} multiple-choice questions (MCQs) from the lecture content below.

For each question return a JSON object with EXACTLY these keys:
- "topic": short topic name (e.g. "Photosynthesis", "Newton's Laws")
- "question_text": the full question text (use **bold** or `code` for emphasis where helpful)
- "option_a": first answer option (text only, no "A." prefix)
- "option_b": second answer option
- "option_c": third answer option
- "option_d": fourth answer option
- "correct_option": the letter of the correct option — exactly one of: "A", "B", "C", or "D"
- "explanation": 1-3 sentence explanation of why the correct answer is right
- "difficulty": "easy", "medium", or "hard"

Requirements:
- Use markdown formatting in question_text and explanation where it improves clarity (**bold**, `code`, etc.)
- All 4 options must be plausible. Only one is correct.
- Cover diverse topics from throughout the content.
- Return a JSON ARRAY only. No markdown code blocks, no text before or after.

Lecture Content:
\"\"\"{content[:6000]}\"\"\"

Return format:
[
  {{
    "topic": "Topic Name",
    "question_text": "What is ...?",
    "option_a": "First choice",
    "option_b": "Second choice",
    "option_c": "Third choice",
    "option_d": "Fourth choice",
    "correct_option": "B",
    "explanation": "Because ...",
    "difficulty": "medium"
  }}
]
"""

        try:
            output_text = generate_ai_content(prompt)
        except Exception as e:
            return JsonResponse({"error": f"AI generation failed: {str(e)}", "details": "Check that your GEMINI_API_KEY is valid."}, status=500)

        cleaned = output_text.strip().replace("```json", "").replace("```", "").strip()

        # Robustly find the JSON array
        import re
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1:
            json_str = cleaned[start:end+1]
        else:
            m = re.search(r'\[.*\]', cleaned, flags=re.DOTALL)
            json_str = m.group(0) if m else cleaned

        try:
            mcqs = json.loads(json_str)
        except json.JSONDecodeError as je:
            return Response({"error": "AI returned invalid JSON", "details": str(je), "raw": cleaned[:500]}, status=500)

        saved = []
        for item in mcqs:
            if not isinstance(item, dict):
                continue
            topic = item.get("topic") or "General"
            q = Question.objects.create(
                lecture_note=note,
                topic=topic,
                question_text=item.get("question_text", ""),
                option_a=item.get("option_a", ""),
                option_b=item.get("option_b", ""),
                option_c=item.get("option_c", ""),
                option_d=item.get("option_d", ""),
                correct_option=(item.get("correct_option") or "A").strip().upper()[:1],
                explanation=item.get("explanation", ""),
                difficulty=0.3 if item.get("difficulty") == "easy" else (0.85 if item.get("difficulty") == "hard" else 0.5)
            )
            saved.append(QuestionSerializer(q).data)
            # Ensure TopicMastery exists for this topic
            if note.user:
                TopicMastery.objects.get_or_create(
                    user=note.user,
                    lecture_note=note,
                    topic=topic,
                    defaults={"mastery": 0.30}
                )

        return Response({"generated_count": len(saved), "questions": saved})

    except Exception as e:
        traceback.print_exc()
        return Response({"error": "MCQ generation failed", "details": str(e)}, status=500)







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
        profile, _ = UserProfile.objects.get_or_create(user=user)
        
        # Calculate stats
        total_quizzes = QuizAttempt.objects.filter(user=user).count()
        if total_quizzes == 0:
            # Fallback to unique questions answered if quiz attempts are not fully logged
            total_quizzes = max(0, UserAnswer.objects.filter(user=user).count() // 10)
        
        # Calculate average score directly from UserAnswer for accuracy
        total_answers = UserAnswer.objects.filter(user=user).count()
        if total_answers > 0:
            correct_answers = UserAnswer.objects.filter(user=user, is_correct=True).count()
            average_score = int((correct_answers / total_answers) * 100)
        else:
            average_score = 0
            
        # Get global streak
        streak_obj = UserStreak.objects.filter(user=user, topic__isnull=True).first()
        streak_days = streak_obj.streak if streak_obj else 0
        
        profile_data = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'bio': profile.bio,
            'total_quizzes': total_quizzes,
            'average_score': average_score,
            'streak_days': streak_days,
            'date_joined': user.date_joined.isoformat() if user.date_joined else None,
        }
        
        return Response(profile_data)
    
    elif request.method == 'PUT':
        profile, _ = UserProfile.objects.get_or_create(user=user)
        
        if 'bio' in request.data:
            profile.bio = request.data.get('bio', '')
            profile.save()
            
        if 'first_name' in request.data:
            user.first_name = request.data.get('first_name')
        if 'last_name' in request.data:
            user.last_name = request.data.get('last_name')
        if 'email' in request.data:
            user.email = request.data.get('email')
            
        user.save()
        
        return Response({
            'status': 'success', 
            'message': 'Profile updated successfully',
            'user': {
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
                'bio': profile.bio
            }
        })



class LectureNoteListView(generics.ListAPIView):
    """
    GET /api/lectures/
    List all lecture notes
    """
    serializer_class = LectureNoteSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Return notes for the current user only
        return LectureNote.objects.filter(user=self.request.user).order_by('-created_at')


class LectureNoteDetailView(generics.RetrieveDestroyAPIView):
    """
    GET /api/lectures/<id>/
    Get details of a lecture note (including questions)
    
    DELETE /api/lectures/<id>/
    Delete a lecture note
    """
    serializer_class = LectureNoteSerializer
    permission_classes = [permissions.IsAuthenticated]
    queryset = LectureNote.objects.all()

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        data = serializer.data
        
        # Add questions to the response
        questions = Question.objects.filter(lecture_note=instance)
        question_serializer = QuestionSerializer(questions, many=True)
        data['questions'] = question_serializer.data
        
        return Response(data)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def generate_lecture_study_aids(request, note_id):
    """
    POST /api/lectures/<id>/generate-study-aids/
    Generate AI-powered study notes, formulas, and key points from lecture content.
    Persists results on the LectureNote model.
    """
    try:
        note = get_object_or_404(LectureNote, id=note_id, user=request.user)
        content = note.content or ""

        if not content.strip():
            return Response({"error": "Lecture has no text content to analyze."}, status=400)

        prompt = f"""You are an expert educational AI. Analyze the following lecture content and produce structured study aids.

Lecture Title: {note.title}

Content:
\"\"\"{content[:8000]}\"\"\"

Return a SINGLE valid JSON object with exactly these keys:
{{
  "study_notes": "# Study Notes\\n\\nA comprehensive markdown-formatted set of notes covering all major topics in the lecture. Use headers (##, ###), bullet points, bold for key terms. Write at least 400 words.",
  "formulas": [
    {{
      "name": "Formula or concept name",
      "formula": "The formula, definition, or rule (use LaTeX-style notation where applicable, e.g. $E = mc^2$)",
      "description": "Brief explanation of when/how to apply this formula or concept"
    }}
  ],
  "key_points": [
    "Concise key point 1",
    "Concise key point 2",
    "..."
  ]
}}

Requirements:
- study_notes: Rich markdown text covering all major topics, definitions, examples. Use ## for sections, **bold** for terms.
- formulas: List of 5-15 important formulas, equations, rules, or definitions from the lecture. If no mathematical formulas, list important conceptual rules/laws.
- key_points: List of 8-15 bullet-point summaries that a student should memorize.
- Return ONLY valid JSON, no markdown code blocks or extra text.
"""

        # Generate study aids via Gemini
        try:
            output_text = generate_ai_content(prompt)
            if not output_text or not output_text.strip():
                return Response({"error": "AI returned empty response. Please try again."}, status=500)
        except Exception as e:
            return Response({"error": f"AI generation failed: {str(e)}"}, status=500)

        # Parse response
        cleaned = output_text.strip().replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]

        data = json.loads(cleaned)

        # Validate and normalize
        study_notes = data.get("study_notes", "")
        formulas = data.get("formulas", [])
        key_points = data.get("key_points", [])

        if not isinstance(formulas, list):
            formulas = []
        if not isinstance(key_points, list):
            key_points = []

        # Persist to DB
        note.study_notes = study_notes
        note.formulas = formulas
        note.key_points = key_points
        note.save(update_fields=["study_notes", "formulas", "key_points"])

        return Response({
            "success": True,
            "study_notes": study_notes,
            "formulas": formulas,
            "key_points": key_points,
        })

    except json.JSONDecodeError as e:
        return Response({"error": "Failed to parse AI response", "details": str(e)}, status=500)
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['PUT'])
@permission_classes([permissions.IsAuthenticated])
def update_question(request, question_id):
    """
    Update an existing question
    PUT /api/questions/<id>/update/
    Body: {
        "question_text": "...",
        "option_a": "...",
        "option_b": "...",
        "option_c": "...",
        "option_d": "...",
        "correct_option": "A/B/C/D",
        "explanation": "..."
    }
    """
    try:
        question = get_object_or_404(Question, id=question_id)
        
        # Update fields if provided
        if 'question_text' in request.data:
            question.question_text = request.data['question_text']
        
        if 'option_a' in request.data:
            question.option_a = request.data['option_a']
        if 'option_b' in request.data:
            question.option_b = request.data['option_b']
        if 'option_c' in request.data:
            question.option_c = request.data['option_c']
        if 'option_d' in request.data:
            question.option_d = request.data['option_d']
            
        if 'correct_option' in request.data:
            correct = request.data['correct_option'].strip().upper()
            if correct not in ['A', 'B', 'C', 'D']:
                return Response(
                    {"error": "correct_option must be A, B, C, or D"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            question.correct_option = correct
            
        if 'explanation' in request.data:
            question.explanation = request.data['explanation']
        
        question.save()
        
        return Response({
            "message": "Question updated successfully",
            "question": QuestionSerializer(question).data
        })
        
    except Exception as e:
        traceback.print_exc()
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )





@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def generate_flashcards(request):
    """
    POST /api/flashcards/generate/
    { "note_id": 123, "count": 5 }
    Generates flashcards using Gemini.
    """
    note_id = request.data.get("note_id")
    count = int(request.data.get("count", 5))
    
    note = get_object_or_404(LectureNote, id=note_id)
    
    prompt = f"""
    Generate exactly {count} flashcards from the following text.
    Each flashcard should have a 'front' (concept/question) and 'back' (definition/answer).
    Keep them concise.
    
    Text:
    {note.content[:3000]}
    
    Return ONLY JSON:
    [
      {{ "front": "Concept", "back": "Definition" }}
    ]
    """
    
    try:
        # Revert to robust model (70B) if Mistral failed
        text = generate_ai_content(prompt)
        # text = response["candidates"][0]["content"]["parts"][0]["text"]
        
        # Clean JSON
        # Clean JSON with Regex
        import re
        cleaned = text.strip()
        
        # Try finding the first '[' and last ']'
        start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx+1]
        else:
            # Fallback regex if manual slicing fails
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            m = re.search(r'\[.*\]', cleaned, flags=re.DOTALL)
            if m:
                cleaned = m.group(0)
            
        flashcards = json.loads(cleaned)
        
        return Response({"flashcards": flashcards})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def summarize_lecture(request, note_id):
    """
    Generate AI-powered summary of a lecture with flowchart visualization.
    Uses Gemini to create structured summary with key concepts, definitions, and Mermaid flowchart.
    """
    try:
        user = request.user
        note = get_object_or_404(LectureNote, id=note_id, user=user)
        
        # Build prompt for structured summary with flowchart
        prompt = f"""
You are an expert educational AI. Analyze the following lecture content and create a comprehensive, structured summary.

Lecture Title: {note.title}

Lecture Content:
\"\"\"
{note.content}
\"\"\"

Generate a JSON response with the following structure:

{{
  "overview": "A 2-3 sentence overview of the entire lecture",
  "key_concepts": [
    {{
      "name": "Concept Name",
      "description": "Brief explanation (1-2 sentences)",
      "importance": "high|medium|low"
    }}
  ],
  "definitions": [
    {{
      "term": "Important Term",
      "definition": "Clear definition"
    }}
  ],
  "relationships": "A paragraph explaining how the key concepts relate to each other",
  "flowchart": "Mermaid.js flowchart syntax showing the flow of concepts (use graph TD format)"
}}

Requirements:
- Include 4-8 key concepts
- Include 3-6 important definitions
- Create a clear, logical flowchart that shows the progression and relationships of concepts
- Use proper Mermaid.js syntax for the flowchart (graph TD or graph LR)
- CRITICAL: Enclose ALL node labels in double quotes, e.g., A["Node Label"]
- Do NOT use special characters (like :, -, (, )) in node labels unless they are inside quotes
- Do NOT use semicolons at the end of lines
- Make the flowchart visually clear with proper node labels
- Return ONLY valid JSON, no markdown code blocks
"""

        try:
            # Use centralized utility for robust generation
            # Returns simple text now
            raw_text = generate_ai_content(prompt).strip()
                
        except Exception as e:
            traceback.print_exc()
            return Response({
                "error": "Failed to generate summary with available models",
                "details": str(e)
            }, status=500)
        
        # Clean up markdown code blocks if present
        import re
        cleaned = raw_text.strip()
        
        # Try finding the first '{' and last '}'
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx+1]
        else:
            # Fallback regex if manual slicing fails
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            # Regex find the main object
            m = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
            if m:
                cleaned = m.group(0)
        
        try:
            summary_data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {raw_text[:500]}")
            return Response({
                "error": "Failed to parse AI response",
                "raw_response": raw_text[:500]
            }, status=500)

        # Validate required fields
        required_fields = ['overview', 'key_concepts', 'definitions', 'relationships', 'flowchart']
        for field in required_fields:
            if field not in summary_data:
                summary_data[field] = [] if field in ['key_concepts', 'definitions'] else ""

        return Response({
            "note_id": note.id,
            "note_title": note.title,
            "summary": summary_data
        }, status=200)

    except Exception as e:
        traceback.print_exc()
        return Response({
            "error": str(e)
        }, status=500)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def generate_video_explanation(request):
    """
    Generate a video explanation script + diagram + audio.
    POST: { "text": "Problem description..." } OR { "question_id": 123 }
    """
    from .models import Question
    from .video_utils import run_video_workflow
    
    user = request.user
    text = request.data.get("text")
    question_id = request.data.get("question_id")
    style = request.data.get("style", "cinematic")
    
    if question_id:
        q = get_object_or_404(Question, id=question_id)
        # Use question text, and maybe the answer/explanation if available for better context
        text = f"Question: {q.question_text}\nAnswer: {q.correct_option}\nExplanation: {q.explanation}"
        
    if not text:
        return Response({"error": "Problem text or question_id required"}, status=400)
        
    result = run_video_workflow(text, style=style)
    
    if "error" in result:
        return Response(result, status=500)
        
    return Response(result)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def get_dashboard_stats(request):
    """
    Returns aggregated statistics for the user dashboard.
    """
    try:
        user = request.user
        
        # 1. Total Study Time (Seconds)
        # Sum of time taken in quizzes (UserAnswer)
        total_quiz_seconds = UserAnswer.objects.filter(user=user).aggregate(ModelsSum('time_taken'))['time_taken__sum'] or 0
        
        # Heuristic: Add 15 mins (900s) for each lecture note uploaded/read
        notes_count = LectureNote.objects.filter(user=user).count()
        total_reading_seconds = notes_count * 900
        
        total_seconds = total_quiz_seconds + total_reading_seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        study_time_str = f"{hours}h {minutes}m"

        # 2. Questions Answered
        questions_answered = UserAnswer.objects.filter(user=user).count()
        
        # 3. Topics Mastered (mastery > 0.8)
        topics_mastered = TopicMastery.objects.filter(user=user, mastery__gt=0.8).count()
        
        # 4. Avg Quiz Score
        # Calculate from UserAnswer correctness
        total_attempts = UserAnswer.objects.filter(user=user).count()
        if total_attempts > 0:
            correct_count = UserAnswer.objects.filter(user=user, is_correct=True).count()
            avg_score = int((correct_count / total_attempts) * 100)
        else:
            avg_score = 0
            
        # 5. Streak
        # Get GLOBAL streak
        streak_obj = UserStreak.objects.filter(user=user, topic__isnull=True).first()
        streak = streak_obj.streak if streak_obj else 0
        
        # 6. Subject Mastery (Aggregate by Topic for now as Subject isn't explicit)
        # We will take top 4 topics
        mastery_data = []
        top_masteries = TopicMastery.objects.filter(user=user).order_by('-mastery')[:4]
        for tm in top_masteries:
            mastery_data.append({
                "subject": tm.topic, # Using topic as subject
                "percentage": int(tm.mastery * 100),
                "color": "#137fec" # Default, frontend can rotate
            })
            
        # 7. Weak Topics
        weak_topics = []
        weakness_objs = TopicWeakness.objects.filter(user=user).order_by('-weakness_score')[:3]
        for w in weakness_objs:
             weak_topics.append({
                 "topic": w.topic,
                 "subject": w.lecture_note.title if w.lecture_note else "General",
                 "note_id": w.lecture_note.id if w.lecture_note else None,
                 "accuracy": max(0, 100 - int(w.weakness_score * 20)) # Heuristic accuracy inverse to weakness
             })
             
        # 8. Recent Activity
        recent_activity = []
        # Last 2 answers
        last_answers = UserAnswer.objects.filter(user=user).order_by('-answered_at')[:2]
        for ans in last_answers:
             recent_activity.append({
                 "type": "quiz",
                 "text": f"Practiced {ans.question.topic or 'General'}",
                 "subtext": f"{'Correct' if ans.is_correct else 'Incorrect'} Answer • {ans.answered_at.strftime('%H:%M')}"
             })
        
        # Last upload
        last_upload = LectureNote.objects.filter(user=user).order_by('-created_at').first()
        if last_upload:
             recent_activity.append({
                 "type": "upload",
                 "text": f"Uploaded \"{last_upload.title}\"",
                 "subtext": f"Processed • {last_upload.created_at.strftime('%H:%M')}"
             })

        return Response({
            "study_time": study_time_str,
            "questions_answered": questions_answered,
            "topics_mastered": topics_mastered,
            "avg_score": avg_score,
            "streak": streak,
            "mastery_data": mastery_data,
            "weak_topics": weak_topics,
            "recent_activity": recent_activity
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def get_lectures_by_topics(request):
    """
    Get lecture note IDs that contain the specified topics.
    GET: ?topics=topic1,topic2,topic3
    Returns: { "note_ids": [1, 2, 3] }
    """
    try:
        user = request.user
        topics_param = request.GET.get('topics', '')
        
        if not topics_param:
            return Response({"error": "Topics parameter is required"}, status=400)
        
        topics = [t.strip() for t in topics_param.split(',') if t.strip()]
        
        if not topics:
            return Response({"error": "No valid topics provided"}, status=400)
        
        # Find lecture notes that have questions related to these topics
        # Using Q objects to search for any of the topics
        query = Q()
        for topic in topics:
            query |= Q(question__topic__icontains=topic)
        
        # Get unique lecture note IDs
        note_ids = LectureNote.objects.filter(
            user=user
        ).filter(query).distinct().values_list('id', flat=True)
        
        return Response({
            "note_ids": list(note_ids),
            "topics": topics
        }, status=200)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)



@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def get_weak_topic_explanation(request):
    """
    Generate AI-powered explanation for a weak topic.
    POST: { "topic": "topic_name", "subject": "subject_name" }
    Returns: { "explanation": "...", "resources": [...], "practice_tips": [...] }
    """
    try:
        user = request.user
        topic = request.data.get('topic')
        subject = request.data.get('subject', 'General')
        
        if not topic:
            return Response({"error": "Topic is required"}, status=400)
        
        # Get topic weakness data for context
        weakness = TopicWeakness.objects.filter(user=user, topic=topic).first()
        mastery = TopicMastery.objects.filter(user=user, topic=topic).first()
        
        # Build context for AI
        context = f"Topic: {topic}\nSubject: {subject}\n"
        if weakness:
            context += f"Weakness Score: {weakness.weakness_score}\n"
        if mastery:
            context += f"Current Mastery: {int(mastery.mastery * 100)}%\n"
        
        # Ultra-optimized prompt for fastest response (2-3 seconds)
        prompt = f"""Explain {topic} for a student. Return JSON:
{{"explanation":"2 sentences with example","key_concepts":["3 concepts"],"practice_tips":["3 tips"]}}"""
        
        
        
        try:
            # Use centralized AI utility
            raw_text = generate_ai_content(prompt).strip()
            
            # Clean up markdown code blocks if present
            import re
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            # Regex find the main object
            m = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
            if m:
                cleaned = m.group(0)
            
            explanation_data = json.loads(cleaned)
            
            # Validate required fields (simplified structure)
            required_fields = ['explanation', 'key_concepts', 'practice_tips']
            for field in required_fields:
                if field not in explanation_data:
                    explanation_data[field] = [] if field in ['key_concepts', 'practice_tips'] else ""
            
            
            return Response({
                "topic": topic,
                "subject": subject,
                "data": explanation_data
            }, status=200)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {raw_text[:500]}")
            return Response({
                "error": "Failed to parse AI response",
                "raw_response": raw_text[:500]
            }, status=500)
            
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)



@api_view(['GET', 'PUT'])
@permission_classes([permissions.IsAuthenticated])
def user_profile(request):
    """
    GET: Retrieve user profile with stats
    PUT: Update user profile fields
    """
    user = request.user
    
    # Get or create profile
    profile, created = UserProfile.objects.get_or_create(user=user)
    
    if request.method == 'GET':
        try:
            # Calculate stats
            quiz_attempts = QuizAttempt.objects.filter(user=user)
            total_quizzes = quiz_attempts.count()
            
            if total_quizzes > 0:
                average_score = quiz_attempts.aggregate(
                    avg=Avg(models.F('score') * 100.0 / models.F('total_questions'))
                )['avg'] or 0
            else:
                average_score = 0
            
            # Get streak
            global_streak = UserStreak.objects.filter(user=user, topic__isnull=True).first()
            streak_days = global_streak.streak if global_streak else 0
            
            return Response({
                'bio': profile.bio or '',
                'school': profile.school or '',
                'grade': profile.grade or '',
                'subjects': profile.subjects or [],
                'preferences': profile.preferences or {},
                'total_quizzes': total_quizzes,
                'average_score': round(average_score, 1),
                'streak_days': streak_days
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)
    
    elif request.method == 'PUT':
        try:
            data = request.data
            
            # Update profile fields
            if 'bio' in data:
                profile.bio = data['bio']
            if 'school' in data:
                profile.school = data['school']
            if 'grade' in data:
                profile.grade = data['grade']
            if 'subjects' in data:
                profile.subjects = data['subjects']
            if 'preferences' in data:
                profile.preferences = data['preferences']
            
            profile.save()
            
            return Response({
                'message': 'Profile updated successfully',
                'bio': profile.bio,
                'school': profile.school,
                'grade': profile.grade,
                'subjects': profile.subjects,
                'preferences': profile.preferences
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# STICKY NOTES CRUD
# ─────────────────────────────────────────────────────────────────────────────

@api_view(['GET', 'POST'])
@permission_classes([permissions.IsAuthenticated])
def sticky_notes(request):
    """
    GET  /api/sticky-notes/?lecture_id=<id>   — list sticky notes (optionally filtered by lecture)
    POST /api/sticky-notes/                    — create a new sticky note
    Body: { "title": "...", "content": "...", "color": "#FFF9C4", "lecture_note_id": <optional> }
    """
    user = request.user

    if request.method == 'GET':
        lecture_id = request.query_params.get('lecture_id')
        qs = StickyNote.objects.filter(user=user)
        if lecture_id:
            qs = qs.filter(lecture_note_id=lecture_id)
        data = [{
            'id': n.id,
            'title': n.title,
            'content': n.content,
            'color': n.color,
            'lecture_note_id': n.lecture_note_id,
            'created_at': n.created_at.isoformat(),
            'updated_at': n.updated_at.isoformat(),
        } for n in qs]
        return Response(data)

    elif request.method == 'POST':
        title = request.data.get('title', 'Class Note').strip() or 'Class Note'
        content = request.data.get('content', '')
        color = request.data.get('color', '#FFF9C4')
        lecture_note_id = request.data.get('lecture_note_id')

        note = StickyNote.objects.create(
            user=user,
            title=title,
            content=content,
            color=color,
            lecture_note_id=lecture_note_id if lecture_note_id else None
        )
        return Response({
            'id': note.id,
            'title': note.title,
            'content': note.content,
            'color': note.color,
            'lecture_note_id': note.lecture_note_id,
            'created_at': note.created_at.isoformat(),
            'updated_at': note.updated_at.isoformat(),
        }, status=201)


@api_view(['PUT', 'DELETE'])
@permission_classes([permissions.IsAuthenticated])
def sticky_note_detail(request, note_id):
    """
    PUT    /api/sticky-notes/<id>/  — update title, content, or color
    DELETE /api/sticky-notes/<id>/  — delete sticky note
    """
    note = get_object_or_404(StickyNote, id=note_id, user=request.user)

    if request.method == 'PUT':
        if 'title' in request.data:
            note.title = request.data['title'].strip() or 'Class Note'
        if 'content' in request.data:
            note.content = request.data['content']
        if 'color' in request.data:
            note.color = request.data['color']
        note.save()
        return Response({
            'id': note.id,
            'title': note.title,
            'content': note.content,
            'color': note.color,
            'lecture_note_id': note.lecture_note_id,
            'updated_at': note.updated_at.isoformat(),
        })

    elif request.method == 'DELETE':
        note.delete()
        return Response({'message': 'Deleted successfully'}, status=200)
