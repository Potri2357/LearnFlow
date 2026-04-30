# Exam Preparation API Views
import pdfplumber
import traceback
import json
import os
import google.generativeai as genai
from .ai_utils import generate_ai_content, cached_generate_ai_content
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import permissions, status
from .models import ExamSyllabus, PreviousQuestionPaper, ExamQuestion, ExamConfiguration

import re

def clean_and_parse_json(text):
    """
    Robustly clean and parse JSON from AI response.
    Handles markdown blocks, trailing commas, and comments.
    """
    try:
        # 1. Remove markdown code blocks
        cleaned = text.replace("```json", "").replace("```", "").strip()
        
        # 2. Extract JSON object or array
        start_brace = cleaned.find('{')
        start_bracket = cleaned.find('[')
        
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            # It's an object
            end_idx = cleaned.rfind('}')
            if end_idx != -1:
                cleaned = cleaned[start_brace:end_idx+1]
        elif start_bracket != -1:
            # It's an array
            end_idx = cleaned.rfind(']')
            if end_idx != -1:
                cleaned = cleaned[start_bracket:end_idx+1]
                
        # 3. Try parsing directly
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 4. Fix common issues if direct parsing fails
        try:
            # Remove comments // ... and /* ... */
            cleaned = re.sub(r'//.*?\n|/\*.*?\*/', '', cleaned, flags=re.S)
            # Remove trailing commas
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            return json.loads(cleaned)
        except Exception:
            raise


def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def upload_exam_syllabus(request):
    """
    Upload exam syllabus (text or PDF)
    POST: { "title": "...", "content": "..." } OR { "title": "...", "file": <PDF> }
    """
    try:
        user = request.user
        title = request.data.get('title', '').strip()
        content = request.data.get('content', '').strip()
        file = request.FILES.get('file')
        
        if not title:
            return Response({"error": "Title is required"}, status=400)
        
        # Extract text from PDF if file is provided
        if file:
            if not file.name.endswith('.pdf'):
                return Response({"error": "Only PDF files are supported"}, status=400)
            
            extracted_text = extract_pdf_text(file)
            if not extracted_text:
                return Response({"error": "Failed to extract text from PDF"}, status=400)
            
            content = extracted_text
        
        if not content:
            return Response({"error": "Content or file is required"}, status=400)
        
        # Create syllabus
        syllabus = ExamSyllabus.objects.create(
            user=user,
            title=title,
            content=content,
            file=file if file else None
        )
        
        return Response({
            "id": syllabus.id,
            "title": syllabus.title,
            "content": syllabus.content[:500] + "..." if len(syllabus.content) > 500 else syllabus.content,
            "created_at": syllabus.created_at.isoformat()
        }, status=201)
        
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def upload_previous_papers(request, syllabus_id):
    """
    Upload previous question papers for pattern analysis
    POST: { "files": [<PDF>, <PDF>, ...] }
    """
    try:
        user = request.user
        syllabus = get_object_or_404(ExamSyllabus, id=syllabus_id, user=user)
        
        files = request.FILES.getlist('files')
        if not files:
            return Response({"error": "No files provided"}, status=400)
        
        uploaded_papers = []
        for file in files:
            if not file.name.endswith('.pdf'):
                continue
            
            extracted_text = extract_pdf_text(file)
            if extracted_text:
                paper = PreviousQuestionPaper.objects.create(
                    exam_syllabus=syllabus,
                    file=file,
                    content=extracted_text
                )
                uploaded_papers.append({
                    "id": paper.id,
                    "filename": file.name,
                    "uploaded_at": paper.uploaded_at.isoformat()
                })
        
        return Response({
            "uploaded_count": len(uploaded_papers),
            "papers": uploaded_papers
        }, status=201)
        
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def generate_exam_questions(request, syllabus_id):
    """
    Generate exam questions using AI
    POST: {
        "total_marks": 100,
        "num_questions": 10,
        "mark_distribution": {"2": 5, "5": 3, "10": 2},
        "secure_centum_mode": false
    }
    """
    try:
        user = request.user
        syllabus = get_object_or_404(ExamSyllabus, id=syllabus_id, user=user)
        
        total_marks = int(request.data.get('total_marks', 100))
        num_questions = int(request.data.get('num_questions', 10))
        mark_distribution = request.data.get('mark_distribution', {})
        secure_centum_mode = request.data.get('secure_centum_mode', False)
        
        # Save configuration
        config = ExamConfiguration.objects.create(
            exam_syllabus=syllabus,
            total_marks=total_marks,
            num_questions=num_questions,
            mark_distribution=mark_distribution,
            secure_centum_mode=secure_centum_mode
        )
        
        # Get previous papers for pattern analysis
        previous_papers = PreviousQuestionPaper.objects.filter(exam_syllabus=syllabus)
        has_previous_papers = previous_papers.exists()
        
        # Analyze patterns if previous papers exist
        pattern_analysis = ""
        if has_previous_papers:
            papers_content = "\n\n---\n\n".join([p.content for p in previous_papers[:3]])  # Limit to 3 papers
            
            pattern_prompt = f"""
Analyze the following previous question papers and identify patterns:

{papers_content[:5000]}

Identify:
1. Frequently asked topics
2. Common question formats
3. Mark distribution patterns
4. Important concepts that appear repeatedly

Return a concise analysis (max 300 words).
"""
            
            try:
                try:
                    pattern_analysis = cached_generate_ai_content('exam_pattern_analysis', pattern_prompt, exam_syllabus=syllabus).strip()
                except:
                    pattern_analysis = "Pattern analysis unavailable"
                print(f"Pattern Analysis: {pattern_analysis}")
            except Exception as e:
                print(f"Pattern analysis failed: {e}")
                pattern_analysis = "Pattern analysis unavailable"
        
        # Generate questions
        secure_mode_instruction = ""
        if secure_centum_mode:
            secure_mode_instruction = """
SECURE CENTUM MODE: Generate comprehensive questions that cover ALL important topics in the syllabus.
Include creative, application-based questions that test deep understanding.
Ensure complete coverage for maximum marks.
"""
        
        pattern_instruction = ""
        if has_previous_papers and pattern_analysis:
            pattern_instruction = f"""
PREVIOUS PAPER PATTERNS IDENTIFIED:
{pattern_analysis}

Use these patterns to prioritize question topics and formats.
"""
        
        # Build mark distribution string
        mark_dist_str = "\n".join([f"- {marks} marks: {count} questions" for marks, count in mark_distribution.items()])
        
        question_prompt = f"""
You are an expert exam question generator. Generate {num_questions} exam questions based on the following syllabus.

SYLLABUS:
{syllabus.content}

{pattern_instruction}

{secure_mode_instruction}

REQUIREMENTS:
- Total marks: {total_marks}
- Number of questions: {num_questions}
- Mark distribution:
{mark_dist_str}

- Each question must have:
  * question_text: The question
  * answer: Detailed answer. MUST be comprehensive.
    - Rule: Provide approx 2 bullet points per mark.
    - Highlight KEY CONCEPTS and PHRASES using **bold** markdown.
    - For 5+ marks, include introduction and conclusion.
  * marks: Mark weightage
  * priority: Priority score (1 = most important)
  * topic: Main topic/concept
  * is_from_pattern: true if based on patterns

- Prioritize questions by importance (most important topics first)
- Ensure mark distribution matches the requirements exactly
- ANSWERS MUST BE DETAILED AND HIGHLIGHT KEY POINTS

- ANSWERS MUST BE DETAILED AND HIGHLIGHT KEY POINTS
- NO COMMENTS in the JSON
- NO TRAILING COMMAS

Return ONLY valid JSON array:
[
  {{
    "question_text": "...",
    "answer": "...",
    "marks": 5,
    "priority": 1,
    "topic": "...",
    "is_from_pattern": false
  }}
]
"""
        
        try:
            raw_text = cached_generate_ai_content('generate_exam_questions', question_prompt, exam_syllabus=syllabus).strip()
        except Exception as e:
            raw_text = ""
            print(f"Generation failed: {e}")
            return Response({
                "error": "Failed to generate questions",
                "details": str(e)
            }, status=500)
                
        try:
            questions_data = clean_and_parse_json(raw_text)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {raw_text[:500]}")
            return Response({
                "error": "Failed to parse AI response",
                "details": "The AI model returned an invalid format. Please try again.",
                "raw_response": raw_text[:1000]
            }, status=500)
        
        # Save questions to database
        created_questions = []
        for q_data in questions_data:
            question = ExamQuestion.objects.create(
                exam_syllabus=syllabus,
                question_text=q_data.get('question_text', ''),
                answer=q_data.get('answer', ''),
                marks=int(q_data.get('marks', 1)),
                priority=int(q_data.get('priority', 999)),
                topic=q_data.get('topic', ''),
                is_from_pattern=q_data.get('is_from_pattern', False)
            )
            created_questions.append({
                "id": question.id,
                "question_text": question.question_text,
                "answer": question.answer,
                "marks": question.marks,
                "priority": question.priority,
                "topic": question.topic,
                "is_from_pattern": question.is_from_pattern
            })
        
        return Response({
            "config_id": config.id,
            "questions_generated": len(created_questions),
            "questions": created_questions,
            "pattern_analysis": pattern_analysis if has_previous_papers else None
        }, status=201)
        
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def get_exam_questions(request, syllabus_id):
    """Get all generated questions for a syllabus"""
    try:
        user = request.user
        syllabus = get_object_or_404(ExamSyllabus, id=syllabus_id, user=user)
        
        questions = ExamQuestion.objects.filter(exam_syllabus=syllabus).order_by('priority', '-marks')
        
        questions_data = [{
            "id": q.id,
            "question_text": q.question_text,
            "answer": q.answer,
            "marks": q.marks,
            "priority": q.priority,
            "topic": q.topic,
            "is_from_pattern": q.is_from_pattern
        } for q in questions]
        
        return Response({
            "syllabus_title": syllabus.title,
            "total_questions": len(questions_data),
            "questions": questions_data
        })
        
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['PUT'])
@permission_classes([permissions.IsAuthenticated])
def update_exam_question(request, question_id):
    """
    Update an exam question
    PUT: { "question_text": "...", "answer": "...", "marks": 5 }
    """
    try:
        user = request.user
        question = get_object_or_404(ExamQuestion, id=question_id, exam_syllabus__user=user)
        
        if 'question_text' in request.data:
            question.question_text = request.data['question_text']
        if 'answer' in request.data:
            question.answer = request.data['answer']
        if 'marks' in request.data:
            question.marks = int(request.data['marks'])
        if 'topic' in request.data:
            question.topic = request.data['topic']
        if 'priority' in request.data:
            question.priority = int(request.data['priority'])
        
        question.save()
        
        return Response({
            "id": question.id,
            "question_text": question.question_text,
            "answer": question.answer,
            "marks": question.marks,
            "priority": question.priority,
            "topic": question.topic
        })
        
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['DELETE'])
@permission_classes([permissions.IsAuthenticated])
def delete_exam_question(request, question_id):
    """Delete an exam question"""
    try:
        user = request.user
        question = get_object_or_404(ExamQuestion, id=question_id, exam_syllabus__user=user)
        question.delete()
        
        return Response({"message": "Question deleted successfully"}, status=200)
        
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def list_exam_syllabi(request):
    """List all exam syllabi for the current user"""
    try:
        user = request.user
        syllabi = ExamSyllabus.objects.filter(user=user).order_by('-created_at')
        
        syllabi_data = [{
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "has_questions": s.exam_questions.exists(),
            "question_count": s.exam_questions.count()
        } for s in syllabi]
        
        return Response({"syllabi": syllabi_data})
        
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def generate_exam_strategy(request, syllabus_id):
    """
    Generate exam strategy and schedule
    POST: { "days_remaining": 5, "hours_per_day": 4 }
    """
    try:
        user = request.user
        syllabus = get_object_or_404(ExamSyllabus, id=syllabus_id, user=user)
        
        days_remaining = int(request.data.get('days_remaining', 1))
        hours_per_day = int(request.data.get('hours_per_day', 2))
        
        # Get previous papers for relevance check
        previous_papers = PreviousQuestionPaper.objects.filter(exam_syllabus=syllabus)
        has_previous_papers = previous_papers.exists()
        
        papers_content = ""
        if has_previous_papers:
            papers_content = "\n\n---\n\n".join([p.content for p in previous_papers[:2]])
            
        prompt = f"""
        Act as an expert exam strategist. Create a detailed study plan for the following syllabus:
        
        SYLLABUS:
        {syllabus.content[:5000]}
        
        PREVIOUS PAPERS ANALYSIS:
        {papers_content[:2000] if papers_content else "No previous papers provided."}
        
        CONSTRAINTS:
        - Days remaining: {days_remaining} (If 0, treat as TODAY ONLY)
        - Hours per day: {hours_per_day}
        
        INSTRUCTIONS:
        1. Analyze the syllabus and identify key topics.
        2. If previous papers are provided, prioritize topics that appear frequently.
        3. Create a day-by-day (or hourly if days=0) schedule.
        4. CRITICAL: For each study block, provide:
           - "duration": e.g., "1.5 hours" or "45 mins" (DO NOT give specific start/end times like 9:00-10:30)
           - "main_topic": The broad area of study
           - "subtopics": A list of specific concepts to cover
           - "type": "study" or "break"
        
           - "type": "study" or "break"

NO COMMENTS. NO TRAILING COMMAS.

OUTPUT FORMAT (JSON ONLY):
        {{
            "is_relevant": true,
            "relevance_warning": "string or null",
            "prioritized_topics": ["topic1", "topic2"],
            "strategy": [
                {{
                    "day": 1,
                    "focus": "Theme of the day",
                    "tasks": [
                        {{
                            "duration": "1.5 hours",
                            "main_topic": "Process Management",
                            "subtopics": ["Process States", "PCB", "Context Switching"],
                            "type": "study"
                        }},
                        {{
                            "duration": "15 mins",
                            "main_topic": "Break",
                            "subtopics": ["Rest and Recharge"],
                            "type": "break"
                        }}
                    ]
                }}
            ]
        }}
        """
        
        try:
            raw_text = cached_generate_ai_content('generate_exam_strategy', prompt, exam_syllabus=syllabus)
        except Exception as e:
            # handle error appropriately
            raw_text = ""
            print(f"Strategy generation failed: {e}")
            raise e
        
        # Parse JSON
        # Parse JSON using robust helper
        try:
            strategy_data = clean_and_parse_json(raw_text)
            return Response(strategy_data)
            
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {raw_text[:500]}")
            return Response({
                "error": "Failed to parse AI response",
                "details": "The AI model returned an invalid format. Please try again.",
                "raw_response": raw_text[:1000]
            }, status=500)
            
    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
