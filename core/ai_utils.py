
import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

# Use the API_KEY from .env which we saw was 'sk-or-v1...'
# Or check specifically for OPENROUTER_API_KEY
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")

def generate_ai_content(prompt, model="meta-llama/llama-3.3-70b-instruct:free"):
    """
    Generates content using OpenRouter API (Meta Llama 3.3).
    Returns the generated text string directly.
    """
    if not OPENROUTER_API_KEY:
        raise Exception("OpenRouter API Key not found (OPENROUTER_API_KEY or API_KEY in env)")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000", # Default local
        "X-Title": "LearnFlow",
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        # Optional: Add temperature, max_tokens if needed
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            # Parse OpenAI-compatible response
            content = result['choices'][0]['message']['content']
            return content
        else:
            print(f"OpenRouter Error {response.status_code}: {response.text}")
            raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"AI Generation Error: {e}")
        # Re-raise to let caller handle fallback or error
        raise e
