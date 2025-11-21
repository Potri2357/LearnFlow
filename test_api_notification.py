
import requests
import os
import django
import sys

# Setup Django to get user credentials if needed, or just use hardcoded if we know them
# But better to use the API directly

BASE_URL = "http://localhost:8000"

def test_notification_flow():
    print("1. Logging in...")
    # We need a valid user. Let's assume 'potri' exists or use the first user from DB
    
    # Get a user from DB to know credentials? No, we can't get password.
    # Let's try to create a test user or use a known one.
    # Since I can't know the password of existing users, I'll create a test user.
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    django.setup()
    from django.contrib.auth.models import User
    from core.models import LectureNote
    
    username = "test_notif_user"
    password = "testpassword123"
    email = "test@example.com"
    
    user = User.objects.filter(username=username).first()
    if not user:
        user = User.objects.create_user(username=username, email=email, password=password)
        print(f"Created test user: {username}")
    else:
        # Reset password to ensure we can login
        user.set_password(password)
        user.save()
        print(f"Reset password for user: {username}")
        
    # Ensure we have a note
    note = LectureNote.objects.first()
    if not note:
        print("Error: No lecture notes found in DB. Please upload one first.")
        return

    print(f"Using Note ID: {note.id}")

    # Login via API
    login_url = f"{BASE_URL}/api/auth/login/"
    response = requests.post(login_url, json={"username": username, "password": password})
    
    if response.status_code != 200:
        print(f"Login failed: {response.status_code} - {response.text}")
        return
        
    tokens = response.json()
    access_token = tokens["access"]
    print("Login successful, got token.")
    
    # Call quiz-completed
    print("2. Calling quiz-completed endpoint...")
    headers = {"Authorization": f"Bearer {access_token}"}
    data = {
        "note_id": note.id,
        "score": 8,
        "total": 10
    }
    
    response = requests.post(f"{BASE_URL}/api/quiz-completed/", json=data, headers=headers)
    
    print(f"API Response Status: {response.status_code}")
    print(f"API Response Body: {response.text}")
    
    if response.status_code == 200:
        print("SUCCESS: API call succeeded.")
    else:
        print("FAILURE: API call failed.")

if __name__ == "__main__":
    test_notification_flow()
