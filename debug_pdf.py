
import requests
import os
import django
import sys
from pathlib import Path

# Setup Django environment
sys.path.append(str(Path(__file__).resolve().parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from core.models import LectureNote

def check_pdf_access():
    print("--- Checking Lecture Notes ---")
    notes = LectureNote.objects.all()
    if not notes.exists():
        print("No lecture notes found in DB.")
        return

    for note in notes:
        print(f"\nID: {note.id} | Title: {note.title}")
        if note.file:
            print(f"File Field: {note.file}")
            print(f"File Path (disk): {note.file.path}")
            print(f"File URL (model): {note.file.url}")
            
            # Construct URL manually to test
            test_url = f"http://127.0.0.1:8000{note.file.url}"
            print(f"Testing Access: {test_url}")
            
            try:
                r = requests.head(test_url)
                print(f"HTTP Status: {r.status_code}")
                if r.status_code == 200:
                    print("SUCCESS: File is accessible via HTTP.")
                    print(f"Content-Type: {r.headers.get('Content-Type')}")
                else:
                    print("FAILURE: File not accessible.")
            except Exception as e:
                print(f"Request Error: {e}")
        else:
            print("No file attached.")

if __name__ == "__main__":
    check_pdf_access()
