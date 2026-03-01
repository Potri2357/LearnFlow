import os
import requests
import json
import logging
from .gemini_utils import generate_with_gemini

logger = logging.getLogger(__name__)

def generate_ai_content(prompt, model=None):
    """
    Generates content using Gemini API — the primary (and only) AI provider.
    Returns the generated text string directly.

    The `model` parameter is accepted for API compatibility but ignored;
    Gemini model selection is handled automatically inside generate_with_gemini()
    with its own priority/fallback list.
    """
    try:
        # Call the robust Gemini utility which handles 429s and model fallbacks
        response = generate_with_gemini(prompt)

        # Extract text from standard Gemini JSON response structure
        # typically: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        if "candidates" in response and len(response["candidates"]) > 0:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"]

        raise Exception(f"Unexpected Gemini response format: {response}")

    except Exception as e:
        logger.error(f"AI Generation Error: {e}")
        # Re-raise to let caller handle fallback or error
        raise e
