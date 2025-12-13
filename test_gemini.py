import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
# Testing the URL that failed
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

payload = {
    "contents": [{
        "parts": [{
            "text": "Say hello"
        }]
    }]
}

try:
    print(f"Testing URL: {url.split('?')[0]}")
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS: gemini-1.5-flash is working!")
    elif response.status_code == 404:
        print("\n❌ 404 NOT FOUND: Model or version incorrect.")
        
        # List of variants to try
        variants = [
            "gemini-2.0-flash",
        ]
        
        for v in variants:
            print(f"Trying {v}...")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{v}:generateContent?key={api_key}"
            try:
                r = requests.post(url, json=payload, timeout=10)
                print(f"{v}: {r.status_code}")
                if r.status_code == 200:
                    print(f"✅ FOUND WORKING MODEL: {v}")
                    break
                elif r.status_code == 429:
                     print(f"⚠️ {v} Rate Limited (429)")
            except:
                pass 
    else:
        print(f"\n⚠️ ERROR: Status {response.status_code}")
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
