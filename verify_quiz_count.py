import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_get_note_details(note_id):
    print(f"Testing get_note_details for note {note_id}...")
    try:
        response = requests.get(f"{BASE_URL}/note-details/{note_id}/")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Note {note_id} details: {data}")
            return data.get("question_count")
        else:
            print(f"FAILURE: Status {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def test_get_quiz_questions(note_id, n):
    print(f"Testing get_quiz_questions for note {note_id} with n={n}...")
    try:
        response = requests.get(f"{BASE_URL}/quiz/{note_id}/?n={n}")
        if response.status_code == 200:
            data = response.json()
            questions = data.get("questions", [])
            count = len(questions)
            print(f"SUCCESS: Received {count} questions.")
            if count <= n:
                print("PASS: Question count is within limit.")
            else:
                print(f"FAIL: Received more questions than requested ({count} > {n}).")
            return count
        else:
            print(f"FAILURE: Status {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

if __name__ == "__main__":
    # Assuming note ID 1 exists. If not, we might need to create one or list them first.
    # For now, let's try note ID 1.
    note_id = 1 
    
    count = test_get_note_details(note_id)
    
    if count is not None:
        # Test with valid n
        test_get_quiz_questions(note_id, min(5, count))
        
        # Test with n > count (should return max available)
        test_get_quiz_questions(note_id, count + 5)
