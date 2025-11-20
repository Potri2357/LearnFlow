import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

payload = {
    "contents": [{
        "parts": [{
            "text": "Say hello"
        }]
    }]
}

try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS: Gemini API is working!")
    elif response.status_code == 429:
        print("\n❌ QUOTA EXCEEDED: Still hitting rate limits")
    else:
        print(f"\n⚠️ ERROR: Status {response.status_code}")
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
