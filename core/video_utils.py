
import os
import json
import base64
import zlib
import requests
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .ai_utils import generate_ai_content
import concurrent.futures
import time

# For TTS (Fallback to placeholder if gTTS not installed)
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

# For Video Assembly
try:
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
    HAS_MOVIEPY = True
except ImportError as e:
    HAS_MOVIEPY = False
    print(f"MoviePy not installed: {e}. Video assembly disabled.")

MEDIA_ROOT = "media" # Root for generated assets

def generate_video_script(problem_text):
    """
    Stage 1: Scripting Engine (Deep Dive Style)
    Generates a dialogue-based JSON script for the video.
    """
    prompt = f"""
    You are a podcast producer creating a "Deep Dive Explainer" for a learning app.
    Create a detailed, information-dense dialogue between two hosts.

    TOPIC/PROBLEM: {problem_text}

    Output a JSON object with this EXACT structure:
    {{
      "title": "<Catchy Video Title>",
      "scenes": [
        {{
          "speaker": "Host" | "Expert",
          "text": "<spoken_text>",
          "image_prompt": "<VISUAL_ONLY_DESCRIPTION_FOR_AI_GENERATOR>",
          "slide_headline": "<SHORT_3_5_WORD_TITLE_FOR_SLIDE>",
          "duration_estimate": <seconds>
        }}
      ]
    }}

    GUIDING PRINCIPLES:
    - **Visuals (CRITICAL)**: The `image_prompt` must be a DESCRIPTION OF A PICTURE, not a description of a concept.
      - BAD: "Explain gravity."
      - GOOD: "A surreal digital art of a floating red apple in space, cinematic lighting."
      - GOOD: "A blueprint schematic of a neural network, glowing lines, dark blue background."
      - KEYWORD: Beautiful, 8k, Detailed, Abstract, Cinematic.
    - **Slide Content**: `slide_headline` should be the "Key Takeaway" of that sentence.
    - **Tone**: Professional, Documentary style.
    - **Length**: 4-6 scenes. Focused. Keep it under 45 seconds total generation.
    """
    
    try:
        text = generate_ai_content(prompt)
        # text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        # Clean JSON
        text = text.replace("```json", "").replace("```", "").strip()
        # Find start and end of JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            text = text[start:end]
            
        script_data = json.loads(text)
        return script_data
    except Exception as e:
        # Sanitize error message for Windows console
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"Error generating video script: {error_msg}")
        return None

def fetch_ai_image(prompt, style="cinematic"):
    """
    Fetches an AI-generated image from Pollinations.ai (No key required).
    """
    try:
        # Enhance prompt based on style
        style_prompt = ""
        if style == "cinematic":
            style_prompt = "cinematic lighting, hyperrealistic, 8k, movie scene"
        elif style == "cartoon":
            style_prompt = "vibrant vector art, flat illustration, clean lines, colorful"
        elif style == "sketch":
            style_prompt = "technical pencil sketch, blueprint, diagram style, white on blue"
        
        full_prompt = f"{prompt}, {style_prompt}".replace(" ", "%20")
        url = f"https://image.pollinations.ai/prompt/{full_prompt}?width=1280&height=720&nologo=true"
        
        return download_image(url)
    except Exception as e:
        print(f"Failed to fetch AI image: {e}")
        return None

def generate_kroki_diagram(visual_description):
    """
    Stage 2: Visual Asset Generation (Diagrams)
    Uses AI to convert the description into Mermaid code, then Kroki to render.
    """
    if not visual_description or len(visual_description) < 10:
        return None
        
    # 1. Get Mermaid Code
    mermaid_prompt = f"""
    Convert this visual description into valid Mermaid JS diagram code.
    DESCRIPTION: {visual_description}
    
    Return ONLY the Mermaid code. No markdown, no explanation.
    
    IMPORTANT SYNTAX RULES:
    1. Do NOT use parentheses () or brackets [] inside node labels unless you wrap the label in quotes.
       BAD: id[Text (Detail)]
       GOOD: id["Text (Detail)"]
    2. Do NOT use special characters like :, -, or . inside ID names. Use simplified IDs (A, B, C).
    3. Keep it simple and readable. Top Down (graph TD) preferred.
    
    Example:
    graph TD;
        A["Start (Input)"] --> B["Process"];
    """
    
    try:
        mermaid_code = generate_ai_content(mermaid_prompt).strip()
        mermaid_code = mermaid_code.replace("```mermaid", "").replace("```", "").strip()
        
        # 2. Encode for Kroki
        # Kroki format: https://kroki.io/mermaid/svg/<base64_encoded_compressed_payload>
        
        # Compress
        compressed = zlib.compress(mermaid_code.encode('utf-8'))
        # Base64 url-safe
        encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')
        
        # We request PNG for easy video integration
        kroki_url = f"https://kroki.io/mermaid/png/{encoded}"
        return kroki_url
        
    except Exception as e:
        print(f"Error generating diagram: {e}")
        return None

def generate_audio_assets(script_data, output_dir="media/audio"):
    """
    Stage 1.4: Generate Voiceover with Dual Personas
    Host: US Accent
    Expert: UK/Australia Accent (to distinguish)
    """
    if not script_data or not HAS_GTTS:
        return []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    audio_files = []
    
    for i, scene in enumerate(script_data.get("scenes", [])):
        text = scene.get("text", "")
        speaker = scene.get("speaker", "Host")
        
        if text:
            filename = f"scene_{i+1}_{random.randint(1000,9999)}.mp3"
            filepath = os.path.join(output_dir, filename)
            try:
                # Select accent based on speaker
                tld = 'us'
                if speaker.lower() == "expert":
                    tld = 'co.uk' # British English
                
                tts = gTTS(text=text, lang='en', tld=tld)
                tts.save(filepath)
                audio_files.append({"scene_index": i, "file": filepath, "speaker": speaker})
            except Exception as e:
                print(f"TTS Error for scene {i}: {e}")
                
    return audio_files

def create_cinematic_slide(title, image_path=None, speaker="Host", output_dir="media/images"):
    """
    Composites an image with a 'Glassmorphism' text card overlay.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    width, height = 1280, 720
    
    # 1. Base Image (Background)
    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            # Resize/Crop to fill
            img_ratio = img.width / img.height
            target_ratio = width / height
            if img_ratio > target_ratio:
                # Too wide, crop width
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img.height))
            else:
                # Too tall, crop height
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                img = img.crop((0, top, img.width, top + new_height))
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Darken it slightly for text contrast
            overlay = Image.new('RGB', (width, height), (0, 0, 0))
            img = Image.blend(img, overlay, 0.3) # 30% darkness
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            img = Image.new('RGB', (width, height), (20, 20, 20))
    else:
        # Fallback Gradient
        img = Image.new('RGB', (width, height), (30, 30, 40))
        
    d = ImageDraw.Draw(img, 'RGBA')
    
    # 2. Glass Card Overlay (Bottom Left)
    card_w, card_h = 1100, 200 # Wider, at bottom
    card_x, card_y = 90, 450
    
    # Semi-transparent white/black rect
    if speaker == "Expert":
        card_color = (255, 255, 255, 220) # Bright for Expert
        text_color = (0, 40, 50)
        tag_color = (0, 100, 150)
    else:
        card_color = (10, 10, 20, 210) # Dark for Host
        text_color = (255, 255, 255)
        tag_color = (180, 100, 255)

    # Draw rounded rect (manually or just rect for now)
    d.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], fill=card_color)
    
    # 3. Text
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 55)
        font_tag = ImageFont.truetype("arial.ttf", 30)
    except:
        font_title = ImageFont.load_default()
        font_tag = ImageFont.load_default()
        
    # Speaker Tag
    d.text((card_x + 30, card_y + 30), speaker.upper(), font=font_tag, fill=tag_color)
    
    # Headline (Wrapped)
    # Simple wrap
    words = title.split()
    lines = []
    line = []
    for w in words:
        if len(' '.join(line + [w])) < 35: # Approx char count
            line.append(w)
        else:
            lines.append(' '.join(line))
            line = [w]
    lines.append(' '.join(line))
    
    y_text = card_y + 80
    for line in lines[:2]: # Max 2 lines
        d.text((card_x + 30, y_text), line, font=font_title, fill=text_color)
        y_text += 65
        
    filename = f"slide_{random.randint(1000,9999)}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    return filepath

def download_image(url, output_dir="media/images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            filename = f"diagram_{random.randint(1000,9999)}.png"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(resp.content)
            return filepath
    except Exception as e:
        print(f"Failed to download diagram: {e}")
    return None

def assemble_video(script, audio_files, visual_assets, output_dir="media/video"):
    """
    Stitch assets into an MP4 using MoviePy.
    """
    if not HAS_MOVIEPY:
        return None
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    clips = []
    
    for i, scene in enumerate(script.get("scenes", [])):
        # 1. Get Audio
        audio_info = next((a for a in audio_files if a["scene_index"] == i), None)
        if not audio_info:
            continue
            
        audio_clip = AudioFileClip(audio_info["file"])
        duration = audio_clip.duration + 0.3
        
        # 2. Get Visual (Slide)
        visual_info = next((v for v in visual_assets if v["scene"] == i), None)
        image_path = None
        
        if visual_info and visual_info.get("local_path"):
            image_path = visual_info["local_path"]
        else:
            # Emergency fallback if slide gen failed
            image_path = create_cinematic_slide(scene.get("slide_headline", "Deep Dive"), None, scene.get("speaker"))
            
        # 3. Create Clip with Ken Burns (Zoom)
        try:
            # MoviePy v2 compatibility
            img_clip = ImageClip(image_path).with_duration(duration)
            
            # Simple Zoom: Crop center
            # w, h = img_clip.size
            # img_clip = img_clip.with_effects([vfx.Resize(lambda t: 1 + 0.04 * t)]) # Zoom in 4%
            # Manual resize for older moviepy or safety
            # img_clip = img_clip.resized(lambda t: 1 + 0.04 * t)
            
            img_clip = img_clip.with_audio(audio_clip)
            # Resize ensure 720p
            img_clip = img_clip.resized(height=720)
            
            clips.append(img_clip)
        except Exception as e:
            print(f"Error creating clip for scene {i}: {e}")
            
    if not clips:
        return None
        
    try:
        final_video = concatenate_videoclips(clips, method="compose")
        
        output_filename = f"deep_dive_{random.randint(1000,9999)}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        final_video.write_videofile(
            output_path, 
            fps=24, 
            codec='libx264', 
            audio_codec='aac'
        )
        return output_path
    except Exception as e:
        print(f"Video export failed: {e}")
        with open("video_error.txt", "w") as f:
            f.write(str(e))
        return None

def run_video_workflow(problem_text, style="cinematic"):
    """
    Orchestrates the full Deep Dive Video workflow.
    """
    print("Starting Deep Dive Video Generation...")
    
    # ensure directories exist
    os.makedirs(os.path.join(MEDIA_ROOT, "audio"), exist_ok=True)
    os.makedirs(os.path.join(MEDIA_ROOT, "images"), exist_ok=True)
    os.makedirs(os.path.join(MEDIA_ROOT, "video"), exist_ok=True)
    
    # 1. Script
    script = generate_video_script(problem_text)
    if not script:
        return {"error": "Script generation failed"}
        
    results = {
        "script": script,
        "assets": []
    }
    
    visual_assets = []
    
    # Helpers for parallel execution
    def process_scene_visual(scene_idx, scene_data):
        # 1. Check for diagram
        v_prompt = scene_data.get("image_prompt", scene_data.get("visual_prompt", "")).lower()
        has_diagram = "diagram" in v_prompt or "chart" in v_prompt
        
        image_path = None
        
        if has_diagram:
            print(f"Generating Diagram for Scene {scene_idx+1}...")
            url = generate_kroki_diagram(v_prompt)
            if url:
                image_path = download_image(url, os.path.join(MEDIA_ROOT, "images"))
        
        # 2. If no diagram, fetch AI image
        if not image_path and len(v_prompt) > 3:
             print(f"Fetching AI Image for Scene {scene_idx+1}: {v_prompt[:30]}...")
             image_path = fetch_ai_image(v_prompt, style=style)

        # 3. Composite into Slide
        headline = scene_data.get("slide_headline", scene_data.get("text", "")[:30])
        speaker = scene_data.get("speaker", "Host")
        final_slide_path = create_cinematic_slide(headline, image_path, speaker, os.path.join(MEDIA_ROOT, "images"))
        
        return {"type": "slide", "scene": scene_idx, "local_path": final_slide_path}

    # Parallel Execution
    visual_assets = []
    audio_files = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 1. Start Audio Generation
        audio_future = None
        if HAS_GTTS:
            print("Generating Voiceovers (Parallel)...")
            audio_future = executor.submit(generate_audio_assets, script, os.path.join(MEDIA_ROOT, "audio"))
        
        # 2. Start Visual Generation
        visual_futures = []
        for i, scene in enumerate(script.get("scenes", [])):
            visual_futures.append(executor.submit(process_scene_visual, i, scene))
            
        # 3. Collect Results
        if audio_future:
            audio_files = audio_future.result()
            results["audio_files"] = audio_files
            
        for f in concurrent.futures.as_completed(visual_futures):
            res = f.result()
            if res:
                visual_assets.append(res)
                results["assets"].append(res)

    # 4. Assembly
    video_url = None
    if HAS_MOVIEPY and audio_files:
        print("Assembling video segments...")
        video_path = assemble_video(script, audio_files, visual_assets, os.path.join(MEDIA_ROOT, "video"))
        if video_path:
            fname = os.path.basename(video_path)
            video_url = f"/media/video/{fname}"
            results["video_url"] = video_url
            print(f"Video created: {video_url}")
        else:
            results["video_error"] = "Assembly failed"
    else:
        results["video_error"] = "MoviePy missing or no audio"
        
    return results
