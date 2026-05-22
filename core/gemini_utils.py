import os
import time
import requests
import logging

logger = logging.getLogger(__name__)


def generate_with_gemini(prompt, model_priority=None):
    """
    Generates content using Gemini with smart retry and fallback logic.

    - gemini-2.5-flash is the primary model (confirmed working on free tier)
    - Daily quota exhaustion (RESOURCE_EXHAUSTED) → skip to next model immediately
    - Per-minute rate limiting (other 429s) → wait & retry same model
    - 404 / model gone → skip to next model immediately
    - Timeout / 503 → retry with backoff

    Returns:
        dict: Standard Gemini JSON response.

    Raises:
        Exception: If all models fail.
    """

    if model_priority is None:
        model_priority = [
            "gemini-2.5-flash",      # Primary - confirmed working on free tier
            "gemini-2.0-flash",      # Fallback 1
            "gemini-2.0-flash-001",  # Fallback 2
            "gemini-2.0-flash-lite", # Fallback 3
            "gemini-2.5-pro",        # Fallback 4 - higher quality
        ]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found in environment variables")

    base_url = "https://generativelanguage.googleapis.com/v1beta/models/"

    last_error = None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
        }
    }

    for model_index, model in enumerate(model_priority):
        url = f"{base_url}{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # Retry up to 3 times per model (for transient rate limits only)
        for attempt in range(3):
            try:
                logger.info(f"Calling {model} (attempt {attempt + 1})")
                response = requests.post(url, json=payload, headers=headers, timeout=120)

                if response.status_code == 200:
                    logger.info(f"Success with {model}")
                    return response.json()

                if response.status_code == 429:
                    try:
                        err_data = response.json().get("error", {})
                        err_status = err_data.get("status", "")
                        err_msg = err_data.get("message", "")
                    except Exception:
                        err_status = ""
                        err_msg = ""

                    # RESOURCE_EXHAUSTED = daily quota gone for this model → skip to next
                    if "RESOURCE_EXHAUSTED" in err_status or "quota" in err_msg.lower():
                        logger.warning(f"Daily quota exhausted on {model}. Trying next model.")
                        last_error = f"Quota exhausted on {model}"
                        break  # Exit inner retry loop, go to next model

                    # Per-minute rate limit → wait with exponential backoff then retry same model
                    wait_s = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    logger.warning(f"Rate limited on {model} (attempt {attempt + 1}). Waiting {wait_s}s...")
                    time.sleep(wait_s)
                    last_error = f"Rate limited on {model}"
                    # Continue to next attempt on same model
                    continue

                if response.status_code == 404:
                    logger.warning(f"Model {model} not found (404). Trying next model.")
                    last_error = f"Model {model} not found"
                    break  # Skip to next model

                if response.status_code == 503:
                    wait_s = 5 * (attempt + 1)
                    logger.warning(f"Service unavailable on {model}. Waiting {wait_s}s...")
                    time.sleep(wait_s)
                    last_error = f"503 Service Unavailable on {model}"
                    continue

                if response.status_code == 400:
                    err_msg = ""
                    try:
                        err_msg = response.json().get("error", {}).get("message", response.text[:200])
                    except Exception:
                        err_msg = response.text[:200]
                    logger.error(f"Bad request on {model}: {err_msg}")
                    last_error = f"Bad request on {model}: {err_msg[:100]}"
                    break  # Bad request - trying another model won't help unless prompt is huge

                # Other status codes
                response.raise_for_status()

            except requests.exceptions.ReadTimeout:
                logger.warning(f"Timeout on {model} (attempt {attempt + 1}).")
                last_error = f"Timeout on {model}"
                time.sleep(5)
                continue

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                last_error = f"Connection error: {str(e)[:100]}"
                time.sleep(3)
                break

            except Exception as e:
                err_msg = str(e).encode('ascii', 'replace').decode('ascii')
                logger.error(f"Error with {model}: {err_msg}")
                last_error = err_msg
                break

    raise Exception(
        f"All Gemini models failed. Last error: {last_error}. "
        "If quota exhausted, wait until it resets (usually midnight Pacific Time) "
        "or upgrade your Google AI Studio / Vertex AI plan."
    )
