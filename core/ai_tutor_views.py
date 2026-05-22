import os
import json
import traceback
import pdfplumber

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import permissions, status

from .ai_utils import generate_ai_content
from .exam_views import clean_and_parse_json, extract_pdf_text


CONCEPT_COACH_SYSTEM_PROMPT = """
You are **Concept Coach AI**, an expert personal tutor. Your teaching philosophy is based on the Socratic method:
- NEVER give the full answer immediately.
- Guide the student with targeted questions, hints, and partial steps.
- Confirm their understanding at each stage before moving on.
- Celebrate correct reasoning and gently correct misconceptions.
- Use formulas when needed, formatted as: "Formula: [formula]" on its own line.
- You can break your response with numbered steps (1. 2. 3.) and bullet points (- ).
- Use **bold** for key terms and `code` for variables/equations.
- Always end your response with a guiding question or a prompt that keeps the student thinking.
- Be warm, encouraging, and patient.

When a student says "give me a hint", provide ONE targeted hint — not the answer.
When a student says "show the formula", provide the relevant formula and explain what each variable means.
When a student says "explain differently", use an analogy or a different approach.
When a student says "I got it, next step", confirm their understanding briefly and give the next step.
When the student clearly has the full correct answer, congratulate them and summarize key learnings.

Respond in plain readable text (no JSON wrappers). Use markdown formatting (bold, lists, formulas) for clarity.
"""


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def concept_coach_chat(request):
    """
    Handle a tutoring chat message.
    POST: { "message": "...", "chat_history": [...], "session_id": "..." }
    Returns: { "response": "...", "hints": [...], "suggestions": [...] }
    """
    try:
        message = request.data.get('message', '').strip()
        chat_history = request.data.get('chat_history', [])

        if not message:
            return Response({"error": "Message is required"}, status=400)

        # Build conversation context (last 4 turns to control token usage)
        context_parts = []
        for msg in chat_history[-4:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '').strip()
            if content:
                label = "Student" if role == 'user' else "Concept Coach"
                context_parts.append(f"{label}: {content}")

        context = "\n\n".join(context_parts)

        full_prompt = f"""{CONCEPT_COACH_SYSTEM_PROMPT}

---
CONVERSATION SO FAR:
{context if context else "(This is the start of the conversation)"}

---
Student: {message}

Concept Coach AI (respond now as described above):"""

        raw_response = generate_ai_content(full_prompt)

        if not raw_response:
            return Response({"error": "AI returned empty response"}, status=500)

        # Extract any hints (lines starting with "Hint:" or "💡")
        hints = []
        response_lines = []
        for line in raw_response.split('\n'):
            stripped = line.strip()
            if stripped.lower().startswith('hint:') or stripped.startswith('💡'):
                hint_text = stripped.replace('💡', '').replace('Hint:', '').strip()
                if hint_text:
                    hints.append(hint_text)
            else:
                response_lines.append(line)

        clean_response = '\n'.join(response_lines).strip()

        return Response({
            "response": clean_response,
            "hints": hints,
            "suggestions": []
        }, status=200)

    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)

        # Build a friendly user-facing message based on error type
        error_lower = error_msg.lower()
        if "quota" in error_lower or "resource_exhausted" in error_lower or "429" in error_lower:
            user_message = (
                "⚠️ **AI quota limit reached.**\n\n"
                "The free tier request limit has been hit temporarily. "
                "Please wait **1–2 minutes** and try again — the per-minute quota resets quickly.\n\n"
                "*If this keeps happening, the daily quota may be exhausted (resets at midnight Pacific Time).*"
            )
        elif "api key" in error_lower or "authentication" in error_lower or "invalid" in error_lower:
            user_message = (
                "🔑 **API key issue detected.**\n\n"
                "The Gemini API key appears to be invalid or missing. "
                "Please check the `GEMINI_API_KEY` environment variable in your backend settings."
            )
        elif "timeout" in error_lower or "connection" in error_lower or "network" in error_lower:
            user_message = (
                "🌐 **Network issue.**\n\n"
                "Could not reach the AI service. Please check your internet connection and try again."
            )
        else:
            user_message = (
                f"❌ **AI error:** {error_msg[:200]}\n\n"
                "Please try again in a moment."
            )

        return Response({
            "response": user_message,
            "hints": [],
            "suggestions": [],
            "is_error": True
        }, status=200)  # 200 so frontend renders it as a chat bubble


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def evaluate_assignment(request):
    """
    Evaluate an uploaded assignment against a rubric.
    POST: file (PDF/TXT/DOCX), subject
    """
    try:
        user = request.user
        file = request.FILES.get('file')
        subject = request.data.get('subject', 'General Assignment')

        if not file:
            return Response({"error": "Assignment file is required"}, status=400)

        # Extract text
        content = ""
        if file.name.endswith('.pdf'):
            content = extract_pdf_text(file)
        else:
            try:
                content = file.read().decode('utf-8')
            except Exception:
                return Response({"error": "Unsupported file format or encoding"}, status=400)

        if not content:
            return Response({"error": "Could not extract text from file"}, status=400)

        eval_prompt = f"""
You are an expert academic evaluator. Evaluate the following assignment for the subject: {subject}.

ASSIGNMENT CONTENT:
{content[:10000]}

Evaluate this assignment and provide comprehensive rubric-based feedback in VALID JSON format only.

JSON Structure:
{{
    "overall_score": 85,
    "quality": "Excellent Performance",
    "content_accuracy_score": 90,
    "clarity_logic_score": 82,
    "originality_pass": true,
    "originality_score": 92,
    "originality_insight": "Well-written with original analysis.",
    "top_strengths": [
        "Strong technical vocabulary",
        "Good logical flow"
    ],
    "actionable_suggestions": [
        {{
            "title": "Improve Academic Tone",
            "description": "Consider using more formal transition phrases."
        }}
    ],
    "mastery_track_name": "Adaptive Mastery Track",
    "mastery_completion_percentage": 82
}}
"""
        raw_response = generate_ai_content(eval_prompt)
        parsed_response = clean_and_parse_json(raw_response)

        return Response({
            "content": content[:2000] + ("..." if len(content) > 2000 else ""),
            "evaluation": parsed_response
        }, status=200)

    except Exception as e:
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
