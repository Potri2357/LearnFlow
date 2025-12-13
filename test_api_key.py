import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ No API key found in .env")
    exit(1)

print(f"✓ API Key loaded: {api_key[:15]}...{api_key[-4:]}")
print(f"✓ Key length: {len(api_key)} characters")

# Test with gemini-2.0-flash
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

payload = {
    "contents": [{
        "parts": [{"text": "Say 'API Working'"}]
    }]
}

print("\nTesting API...")
try:
    response = requests.post(url, json=payload, timeout=15)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS! New API key is working!")
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Response: {text}")
    elif response.status_code == 429:
        print("⚠️ 429 Rate Limited - Key is valid but quota exceeded")
    elif response.status_code == 400:
        print("❌ 400 Bad Request - Check API key format")
        print(f"Response: {response.text[:200]}")
    else:
        print(f"❌ Error {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")
