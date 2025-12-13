import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"

try:
    print(f"Listing models from: {url.split('?')[0]}")
    resp = requests.get(url)
    if resp.status_code == 200:
        models = resp.json().get('models', [])
        import json
        with open('models.json', 'w') as f:
            json.dump([m['name'] for m in models], f, indent=2)
        print("Saved to models.json")
    else:
        print(f"Error: {resp.status_code} {resp.text}")
except Exception as e:
    print(e)
