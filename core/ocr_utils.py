
import re
import os
import logging
from collections import Counter
import numpy as np
import threading

# Try imports for OCR (may fail if not installed)
try:
    import pytesseract
    from pdf2image import convert_from_path, convert_from_bytes
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False
    from PIL import Image
    HAS_OCR = True
    
    # Windows: Set Tesseract Path if not in PATH
    if os.name == 'nt':
        default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(default_path) and not pytesseract.pytesseract.tesseract_cmd.endswith('tesseract.exe'):
            pytesseract.pytesseract.tesseract_cmd = default_path
            
except ImportError:
    HAS_OCR = False
    print("OCR dependencies (pytesseract, pdf2image, opencv, pillow) not fully installed.")

from .ai_utils import generate_ai_content

logger = logging.getLogger(__name__)

# =========================================================
# PHASE 0: INPUT CLASSIFICATION
# =========================================================

def is_scanned_pdf(text_content):
    """
    Heuristic: If extracted text is very short relative to file size 
    or mostly garbage, assume scanned.
    """
    if not text_content or len(text_content.strip()) < 50:
        return True
    
    # Check density of valid words vs symbols
    # ... implementation detail ...
    return False

# =========================================================
# PHASE 1: OCR (Only for Scanned Inputs)
# =========================================================

def preprocess_image(image):
    """
    Phase 1.2: Preprocess images (Grayscale -> Threshold -> Noise Removal)
    """
    if not HAS_OCR: return image
    minutes = None
    
    # Phase 1.2: Preprocess images (Grayscale -> Threshold -> Noise Removal)
    if not HAS_OCR: return image
    
    # 1. Fallback if CV2 missing (Use PIL)
    if not HAS_CV2:
        return image.convert('L').point(lambda x: 0 if x < 128 else 255, '1')

    # Convert PIL to cv2
    img = np.array(image)
    
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img 
        
    # 2. Thresholding (Otsu)
    try:
         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
         return Image.fromarray(thresh)
    except:
         return image.convert('L')

def perform_ocr(file_bytes=None, file_path=None):
    """
    Phase 1.3: Run OCR per page and merge.
    """
    if not HAS_OCR:
        # Return None to indicate OCR is not possible, let caller decide (e.g. reject upload)
        logger.warning("OCR requested but dependencies missing.")
        return None
        
    text_output = ""
    try:
        # Convert PDF to Images
        if file_path:
            images = convert_from_path(file_path)
        else:
            images = convert_from_bytes(file_bytes)
            
        for i, img in enumerate(images):
            # Preprocess
            processed_img = preprocess_image(img)
            
            # Tesseract
            page_text = pytesseract.image_to_string(processed_img)
            text_output += f"\n\n{page_text}"
            
    except Exception as e:
        logger.error(f"OCR Failed: {e}")
        return f"[OCR Failed: {e}]"
        
    return text_output

# =========================================================
# PHASE 2: STRUCTURAL CLEANUP
# =========================================================

def clean_structural_noise(text):
    """
    Phase 2: Remove headers, footers, page numbers, random symbols.
    """
    # Pre-clean: Fix hyphenated words at end of lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # 1. Drop short lines (likely page numbers or noise)
        if len(line) < 4:
            continue
            
        # 2. Drop Page numbers pattern "Page 1 of 10" or just digit
        if re.match(r'^Page \d+', line, re.IGNORECASE) or re.match(r'^\d+$', line):
            continue
            
        # 3. Remove repeated separators
        if re.match(r'^[_=-]{3,}$', line):
            continue
            
        # 4. Remove lines with high density of symbols (OCR noise)
        symbol_count = len(re.findall(r'[^a-zA-Z0-9\s]', line))
        if len(line) > 0 and (symbol_count / len(line)) > 0.5:
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

# =========================================================
# PHASE 3: SENTENCE SEGMENTATION
# =========================================================

def segment_sentences(text):
    """
    Phase 3: Split into semantic sentence units.
    Uses regex for independence from heavy NLTK/Spacy.
    """
    # Protect common abbreviations (Dr., e.g., i.e.)
    text = text.replace("e.g.", "e_g_").replace("i.e.", "i_e_").replace("Dr.", "Dr_")
    
    # Split on period/question/exclamation followed by space/end, but not preceded by known abbreviations
    # Simple regex split
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # Restore abbreviations
    sentences = [s.replace("e_g_", "e.g.").replace("i_e_", "i.e.").replace("Dr_", "Dr.") for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]

# =========================================================
# PHASE 4: IMPORTANCE SCORING
# =========================================================

IMPORTANT_INDICATORS = [
    "is defined as", "refers to", "means", # Definitions
    "steps", "algorithm", "procedure", "process", # Processes
    "architecture", "model", "framework", "structure", # Concepts
    "=", "->", "%", "O(", "formula", "equation", # Math
    "because", "therefore", "results in", "leads to", # Cause-Effect
    "key", "important", "significant" # Signals
]

NOISE_INDICATORS = [
    "example", "imagine", "suppose", "let's say", # Examples
    "chapter", "unit", "module", "syllabus", # Meta
    "hello", "welcome", "objective", "in semantic terms" # Filler
]

def filter_important_content(sentences):
    """
    Phase 4: Rule-based importance filtering.
    """
    kept_sentences = []
    
    for s in sentences:
        s_lower = s.lower()
        
        # 1. Skip strictly noise
        if any(bad in s_lower for bad in NOISE_INDICATORS):
            continue
            
        # 2. Keep if strictly important
        if any(good in s_lower for good in IMPORTANT_INDICATORS):
            kept_sentences.append(s)
            continue
            
        # 3. Keep if decent length and seems academic (has generic academic words)
        # Fallback: keep top 50% by length if unsure? No, "Structure First".
        # Let's keep if it has > 10 words (likely a full thought)
        if len(s.split()) > 8:
            kept_sentences.append(s)
            
    return " ".join(kept_sentences)

# =========================================================
# PHASE 5: AI SEMANTIC COMPRESSION
# =========================================================

def compress_content_with_ai(text):
    """
    Phase 5: Send filtered text to AI to extract ONLY main content.
    """
    # Chunking slightly to fit context window if needed, but assuming filtered text is manageable.
    # Limit to e.g. 15k chars for prompt safety
    
    chunk = text[:15000] 
    
    prompt = f"""
    Analyze the text below and extract highly dense "Exam-Ready" notes.
    
    STRICT RULES:
    1. EXTRACT: Definitions, Formulas, Steps, Algorithms, and Causal Relationships (A -> B).
    2. IGNORE: Anecdotes, conversational filler, "Welcome to the course", "In this chapter".
    3. FORMAT: Markdown bullet points with bold keywords.
    4. ACCURACY: Do not hallucinate. Use only provided text.
    5. DENSITY: High. Make every word count.
    
    Input Text:
    {chunk}
    """
    
    try:
        # Use default model (Llama 70B) for high-quality semantic compression
        compressed = generate_ai_content(prompt)
        return compressed
    except Exception as e:
        logger.error(f"AI Compression Failed: {e}")
        return text # Fallback to filtered text

# =========================================================
# MAIN PIPELINE
# =========================================================

def process_document_pipeline(file_obj=None, file_path=None, extracted_text_if_digital=None):
    """
    Orchestrates the full pipeline.
    """
    # Phase 0 & 1: Source Identification & OCR
    raw_text = extracted_text_if_digital or ""
    
    if is_scanned_pdf(raw_text):
        print("Detected Scanned Document. Attempting OCR...")
        if HAS_OCR:
            raw_text = perform_ocr(file_bytes=file_obj.read() if file_obj else None, file_path=file_path)
        else:
            print("OCR Unavailable. Content may be empty.")
    
    # Phase 2: Structural Cleanup
    clean_text = clean_structural_noise(raw_text)
    
    # Phase 3: Segmentation
    sentences = segment_sentences(clean_text)
    
    # Phase 4: Importance Scoring
    filtered_text = filter_important_content(sentences)
    
    # Phase 5: Semantic Compression
    final_content = compress_content_with_ai(filtered_text)
    
    return final_content
