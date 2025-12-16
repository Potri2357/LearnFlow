import os
import time
import requests
import google.generativeai as genai
import logging

# Configure logging
logger = logging.getLogger(__name__)

def generate_with_gemini(prompt, model_priority=None):
    """
    Generates content using Gemini with automatic model fallback and retry logic.
    
    Args:
        prompt (str): The prompt text to send to the model.
        model_priority (list): Optional list of models to try in order. 
                               Defaults to [gemini-1.5-flash, gemini-1.5-flash-8b, gemini-2.0-flash-exp, gemini-1.5-pro]
    
    Returns:
        dict: The JSON response from the API (standard Gemini format).
        
    Raises:
        Exception: If all models fail or API key is missing.
    """
    
    if model_priority is None:
        model_priority = [
            "gemini-2.5-flash",    # Found in list
            "gemini-3-pro-exp",    # Found in list
            "deepthink-exp-05-20", # Found in list
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash"
        ]
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found in environment variables")
        
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/"
    
    last_error = None
    
    # Try the entire chain up to 3 times if global 429s occur
    max_global_retries = 3
    
    for attempt in range(max_global_retries):
        for i, model in enumerate(model_priority):
            url = f"{base_url}{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            
            # Consistent payload structure
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            
            try:
                # Short timeout per model to allow trying multiple
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    # Success!
                    return response.json()
                
                # Error Handling
                if response.status_code == 429:
                    print(f"Limit hit on {model} (429). Waiting 5s before switching...")
                    time.sleep(5) # Backoff
                    last_error = f"429 Rate Limit on {model}"
                    continue
                
                if response.status_code == 404:
                    print(f"Model {model} unavailable (404). Switching...")
                    last_error = f"404 Model {model} not found"
                    continue
                    
                if response.status_code == 503:
                    print(f"Service unavailable on {model} (503). Switching...")
                    time.sleep(2)
                    last_error = f"503 Service Unavailable on {model}"
                    continue
    
                # Check if it's a 400 Bad Request (invalid argument) -> don't retry, just raise or log
                if response.status_code == 400:
                    print(f"Bad Request on {model}: {response.text}")
                    last_error = f"400 Bad Request on {model}: {response.text}"
                    continue
                    
                # Other errors: raise to catch below
                response.raise_for_status()
                
            except requests.exceptions.ReadTimeout:
                 print(f"Timeout on {model}. Switching...")
                 last_error = f"Timeout on {model}"
                 continue
            except Exception as e:
                # Sanitize error message for Windows console
                err_msg = str(e).encode('ascii', 'replace').decode('ascii')
                print(f"Error with {model}: {err_msg}")
                last_error = err_msg
                continue
        
        # If we finished the loop without success, verify if we should global retry
        # Only global retry if we hit rate limits? For now, we just retry the whole chain once more.
        if attempt < max_global_retries - 1:
            print(f"All models failed attempt {attempt+1}. Retrying chain in 5s...")
            time.sleep(5)
    
    # If we get here, all models failed after retries
    raise Exception(f"All Gemini models failed. Last error: {last_error}")
    
    # If we get here, all models failed
    raise Exception(f"All Gemini models failed. Last error: {last_error}")
