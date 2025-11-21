import os
import django
import requests
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from core.models import LectureNote
from rest_framework.test import APIRequestFactory
from core.views import generate_study_plan

def test_study_plan():
    try:
        note = LectureNote.objects.first()
        if not note:
            print("No lecture notes found.")
            return

        print(f"Testing with note ID: {note.id}")
        
        factory = APIRequestFactory()
        request = factory.post('/api/study-plan/', {'note_id': note.id}, format='json')
        
        response = generate_study_plan(request)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            data = response.data
            print("Plan length:", len(data.get('plan', '')))
            print("Sections:", data.get('plan_sections', {}).keys())
        else:
            print("Failed!")
            print(f"Status Code: {response.status_code}")
            print(response.content.decode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    with open("reproduce_output.txt", "w") as f:
        sys.stdout = f
        sys.stderr = f
        test_study_plan()
