"""
One-time script to clean existing question options in the database.
This removes letter prefixes (A), B), C), D)) from all existing options.

Run this once after deploying the clean_option_text fix:
python manage.py shell < clean_existing_options.py
"""

from core.models import Question
import re

def clean_option_text(text):
    """Remove letter prefixes from option text"""
    if not text:
        return ""
    cleaned = re.sub(r'^[A-D]\)\s*', '', str(text).strip(), flags=re.IGNORECASE)
    return cleaned.strip()

# Get all questions
questions = Question.objects.all()
updated_count = 0

print(f"Found {questions.count()} questions to check...")

for q in questions:
    updated = False
    
    # Clean each option
    if q.option_a:
        cleaned_a = clean_option_text(q.option_a)
        if cleaned_a != q.option_a:
            q.option_a = cleaned_a
            updated = True
    
    if q.option_b:
        cleaned_b = clean_option_text(q.option_b)
        if cleaned_b != q.option_b:
            q.option_b = cleaned_b
            updated = True
    
    if q.option_c:
        cleaned_c = clean_option_text(q.option_c)
        if cleaned_c != q.option_c:
            q.option_c = cleaned_c
            updated = True
    
    if q.option_d:
        cleaned_d = clean_option_text(q.option_d)
        if cleaned_d != q.option_d:
            q.option_d = cleaned_d
            updated = True
    
    if updated:
        q.save()
        updated_count += 1
        print(f"Updated question {q.id}: {q.question_text[:50]}...")

print(f"\nDone! Updated {updated_count} questions.")
