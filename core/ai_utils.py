import os
import time
import logging
import requests

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# PROVIDER REGISTRY
# Add your API keys to the .env file:
#   GEMINI_API_KEY=AIza...
#   GROQ_API_KEY=gsk_...
#   OPENROUTER_API_KEY=sk-or-...
# Any provider whose key is missing is automatically skipped.
# ────────────────────────────────────────────────────────────────────────────────


def _call_gemini(prompt: str, timeout: int = 120, max_tokens: int = 1200) -> str:
    """Gemini REST API — tries gemini-2.5-flash, then 2.0-flash."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    base = "https://generativelanguage.googleapis.com/v1beta/models/"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": max_tokens},
    }

    for model in models:
        url = f"{base}{model}:generateContent?key={api_key}"
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                if text and text.strip():
                    logger.info(f"[Gemini] Success with {model}")
                    return text
            elif r.status_code == 429:
                err = r.json().get("error", {})
                if "RESOURCE_EXHAUSTED" in err.get("status", ""):
                    logger.warning(f"[Gemini] Quota exhausted on {model}, trying next")
                    continue  # skip to next model
                else:
                    logger.warning(f"[Gemini] Rate limited on {model}, waiting 10s")
                    time.sleep(10)
                    # Retry once
                    r2 = requests.post(url, json=payload, timeout=timeout)
                    if r2.status_code == 200:
                        data = r2.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        if text and text.strip():
                            return text
                    continue
            elif r.status_code == 404:
                logger.warning(f"[Gemini] Model {model} not found, trying next")
                continue
            else:
                logger.warning(f"[Gemini] {r.status_code} on {model}")
                continue
        except requests.exceptions.Timeout:
            logger.warning(f"[Gemini] Timeout on {model}")
            continue
        except Exception as e:
            logger.warning(f"[Gemini] Error on {model}: {e}")
            continue

    raise Exception("Gemini: all models exhausted or quota exceeded")


def _call_groq(prompt: str, timeout: int = 60, max_tokens: int = 1200) -> str:
    """
    Groq — ultra-fast inference, 14,400 free requests/day.
    Free models: llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it, mixtral-8x7b-32768
    Get key at: https://console.groq.com/
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    models = [
        "llama-3.3-70b-versatile",   # best quality, generous quota
        "llama3-70b-8192",            # alias
        "llama-3.1-8b-instant",       # fastest fallback
        "gemma2-9b-it",               # Google Gemma via Groq
        "mixtral-8x7b-32768",         # Mixtral fallback
    ]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                if text and text.strip():
                    logger.info(f"[Groq] Success with {model}")
                    return text
            elif r.status_code == 429:
                err = r.json().get("error", {})
                err_type = err.get("type", "")
                if "daily" in err_type or "tokens_exceeded" in err_type:
                    logger.warning(f"[Groq] Daily limit on {model}, trying next")
                    continue
                else:
                    # Per-minute limit — short wait
                    logger.warning(f"[Groq] Rate limited on {model}, waiting 8s")
                    time.sleep(8)
                    r2 = requests.post(url, json=payload, headers=headers, timeout=timeout)
                    if r2.status_code == 200:
                        text = r2.json()["choices"][0]["message"]["content"]
                        if text and text.strip():
                            return text
                    continue
            elif r.status_code == 404:
                logger.warning(f"[Groq] Model {model} not available, trying next")
                continue
            else:
                logger.warning(f"[Groq] {r.status_code} on {model}: {r.text[:100]}")
                continue
        except requests.exceptions.Timeout:
            logger.warning(f"[Groq] Timeout on {model}")
            continue
        except Exception as e:
            logger.warning(f"[Groq] Error on {model}: {e}")
            continue

    raise Exception("Groq: all models failed or quota exceeded")


def _call_openrouter(prompt: str, timeout: int = 90, max_tokens: int = 1200) -> str:
    """
    OpenRouter — single key routes to many free models.
    Free models: deepseek/deepseek-r1, meta-llama/llama-3.3-70b, mistralai/mistral-7b, etc.
    Get key at: https://openrouter.ai/
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    models = [
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "google/gemma-3-27b-it:free",
    ]

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://learnflow.app",
        "X-Title": "LearnFlow AI",
    }

    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if text and text.strip() and text != "":
                        logger.info(f"[OpenRouter] Success with {model}")
                        return text
                    else:
                        logger.warning(f"[OpenRouter] Empty response from {model}")
                        continue
            elif r.status_code == 429:
                logger.warning(f"[OpenRouter] Rate limited on {model}, trying next")
                time.sleep(5)
                continue
            elif r.status_code in (404, 400):
                logger.warning(f"[OpenRouter] {r.status_code} on {model}, trying next")
                continue
            else:
                logger.warning(f"[OpenRouter] {r.status_code} on {model}: {r.text[:100]}")
                continue
        except requests.exceptions.Timeout:
            logger.warning(f"[OpenRouter] Timeout on {model}")
            continue
        except Exception as e:
            logger.warning(f"[OpenRouter] Error on {model}: {e}")
            continue

    raise Exception("OpenRouter: all models failed or quota exceeded")


def _call_gemini_sdk(prompt: str, max_tokens: int = 1200) -> str:
    """
    Secondary Gemini path via official SDK — separate code path from REST API.
    Used as last Gemini resort before giving up entirely.
    """
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    sdk_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

    for model_name in sdk_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = response.text
            if text and text.strip():
                logger.info(f"[Gemini-SDK] Success with {model_name}")
                return text
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "429" in err or "resource_exhausted" in err:
                logger.warning(f"[Gemini-SDK] Quota on {model_name}")
                continue
            logger.warning(f"[Gemini-SDK] Error on {model_name}: {e}")
            continue

    raise Exception("Gemini SDK: all models failed")


# ────────────────────────────────────────────────────────────────────────────────
# PROVIDER PIPELINE — tries providers in order, skips unconfigured ones
# ────────────────────────────────────────────────────────────────────────────────

_PROVIDERS = [
    ("Gemini",      _call_gemini,      "GEMINI_API_KEY"),
    ("Groq",        _call_groq,        "GROQ_API_KEY"),
    ("OpenRouter",  _call_openrouter,  "OPENROUTER_API_KEY"),
    ("Gemini-SDK",  _call_gemini_sdk,  "GEMINI_API_KEY"),   # Second Gemini attempt via SDK
]


TOKEN_BUDGETS = {
    # Canonical budgets
    'summarize': 1200,
    'flashcard': 120,
    'mcq': 180,
    'coach': 600,
    'plan': 1500,
    # Action names used by views/exam_views
    'summarize_lecture': 1200,
    'generate_flashcards': 120,
    'generate_mcqs': 180,
    'generate_study_aids': 1200,
    'study_plan': 1500,
    'detect_subject': 80,
    'exam_pattern_analysis': 800,
    'generate_exam_questions': 1400,
    'generate_exam_strategy': 1500,
}


def generate_ai_content(prompt: str, model=None, max_tokens: int = None, action_type: str = None) -> str:
    """
    Generates AI text content using a multi-provider pipeline.

    Provider order:
    1. Gemini (REST)      — gemini-2.5-flash → 2.0-flash → 2.0-flash-lite
    2. Groq               — llama-3.3-70b → llama-3.1-8b → gemma2 → mixtral
    3. OpenRouter         — deepseek-r1 → llama-3.3-70b → mistral-7b → qwen
    4. Gemini SDK         — second attempt via official SDK

    Any provider whose API key is missing in the .env is silently skipped.
    Returns the first successful text response.

    Add keys to your .env file:
        GEMINI_API_KEY=AIza...        (https://aistudio.google.com)  [already set]
        GROQ_API_KEY=gsk_...          (https://console.groq.com)     [FREE, recommended]
        OPENROUTER_API_KEY=sk-or-...  (https://openrouter.ai)        [FREE tier available]

    The `model` param is accepted for compatibility but ignored; each provider
    handles its own model selection internally.

    Returns:
        str: The generated text.

    Raises:
        Exception: When all configured providers fail.
    """
    errors = []
    at_least_one_configured = False

    # Determine token budget
    if max_tokens is None and action_type:
        max_tokens = TOKEN_BUDGETS.get(action_type, 1200)

    for provider_name, provider_fn, key_env in _PROVIDERS:
        # Skip if key not set (except SDK which shares Gemini key)
        if not os.environ.get(key_env, ""):
            logger.debug(f"[{provider_name}] Skipping — {key_env} not configured")
            continue

        at_least_one_configured = True
        try:
            # Pass max_tokens where provider supports it
            try:
                result = provider_fn(prompt, max_tokens=max_tokens)
            except TypeError:
                result = provider_fn(prompt)
            if result and result.strip():
                return result
            else:
                errors.append(f"{provider_name}: returned empty response")
        except ValueError as e:
            # Key not set — skip silently
            continue
        except Exception as e:
            err_msg = str(e)
            errors.append(f"{provider_name}: {err_msg[:120]}")
            logger.warning(f"[{provider_name}] Failed: {err_msg[:120]}")
            continue

    if not at_least_one_configured:
        raise Exception(
            "No AI providers configured. Please add at least GEMINI_API_KEY to your .env file."
        )

    # Build user-friendly error
    all_errors = "\n• ".join(errors)
    is_quota = any(
        kw in " ".join(errors).lower()
        for kw in ["quota", "resource_exhausted", "exhausted", "rate limit", "429"]
    )

    if is_quota:
        raise Exception(
            "AI quota limit reached on all providers. "
            "Please wait a few minutes and try again, or add a GROQ_API_KEY / OPENROUTER_API_KEY "
            "to your .env for additional free-tier providers.\n"
            f"Details:\n• {all_errors}"
        )

    raise Exception(
        f"All AI providers failed:\n• {all_errors}"
    )

def cached_generate_ai_content(action_type: str, prompt: str, model=None, ttl_hours: int = 24, lecture_note=None, exam_syllabus=None) -> str:
    """
    Wrapper around generate_ai_content that caches responses based on (lecture_note or exam_syllabus, action_type)
    for `ttl_hours`. Useful for saving tokens on repeated requests (like generating a summary or quiz).
    """
    from core.models import AIResponseCache
    from django.utils import timezone
    from datetime import timedelta
    import logging

    logger = logging.getLogger(__name__)

    if not lecture_note and not exam_syllabus:
        # No cache scope available; still apply action-specific token budget.
        max_tokens = TOKEN_BUDGETS.get(action_type, 1200)
        return generate_ai_content(prompt, model=model, max_tokens=max_tokens, action_type=action_type)

    # Try to find a valid cache
    cache_qs = AIResponseCache.objects.filter(
        lecture_note=lecture_note,
        exam_syllabus=exam_syllabus,
        action_type=action_type
    )
    if cache_qs.exists():
        cache_entry = cache_qs.first()
        # Check if expired
        if cache_entry.updated_at >= timezone.now() - timedelta(hours=ttl_hours):
            try:
                if isinstance(cache_entry.response_data, dict) and "text" in cache_entry.response_data:
                    return cache_entry.response_data["text"]
                elif isinstance(cache_entry.response_data, str):
                    return cache_entry.response_data
            except Exception as e:
                logger.warning(f"Failed to read cache for {action_type}: {e}")
                pass
    
    # Not cached or expired, generate new content
    # Map action_type to token budget
    max_tokens = TOKEN_BUDGETS.get(action_type, 1200)
    response_text = generate_ai_content(prompt, model=model, max_tokens=max_tokens, action_type=action_type)
    
    # Save to cache
    try:
        AIResponseCache.objects.update_or_create(
            lecture_note=lecture_note,
            exam_syllabus=exam_syllabus,
            action_type=action_type,
            defaults={
                'response_data': {"text": response_text}
            }
        )
    except Exception as e:
        logger.warning(f"Failed to save cache for {action_type}: {e}")

    return response_text
