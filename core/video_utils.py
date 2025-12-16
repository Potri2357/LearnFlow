
import os
import json
import base64
import zlib
import requests
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .ai_utils import generate_ai_content

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
    You are a podcast producer creating a "Deep Dive" segment for a learning app.
    Create a lively, conversational EQUATION/CONCEPT BREAKDOWN between two hosts:
    1. **Host (US Accent)**: Energetic, curious, sets the stage, asks the "dumb" questions that students are thinking.
    2. **Expert (British Accent)**: Knowledgeable, calm, explanatory, uses analogies.

    TOPIC/PROBLEM: {problem_text}

    Output a JSON object with this EXACT structure:
    {{
      "title": "<Catchy Video Title>",
      "scenes": [
        {{
          "speaker": "Host" | "Expert",
          "text": "<spoken_text>",
          "visual_prompt": "<description_for_diagram_generator_or_slide>",
          "duration_estimate": <seconds>
        }}
      ]
    }}

    GUIDING PRINCIPLES:
    - **Conversational Tone**: Use natural language, "Um," "So," "Wait," "Exactly!". Make it sound like a real chat.
    - **Structure**:
      1. **Intro**: Host introduces the "mystery" or challenge.
      2. **Breakdown**: Expert explains, Host interrupts with clarification questions.
      3. **Analogy**: Expert uses a real-world analogy (CRITICAL).
      4. **Summary**: Host summarizes what they learned.
    - **Visuals**:
      - For the 'visual_prompt', describe simpler, clear slides or diagrams.
      - "Title card: <Text>", "Split screen: <Text>", "Diagram of <Concept>".
    - **Length**: Keep it concise. Around 6-10 exchanges max. 45-60 seconds total.
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

def generate_kroki_diagram(visual_description):
    """
    Stage 2: Visual Asset Generation (Diagrams)
    Uses Gemini to convert the description into Mermaid code, then Kroki to render.
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
        # mermaid_code = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
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

def create_text_slide(title, subtitle=None, speaker="Host", output_dir="media/images"):
    """
    Creates a text slide.
    Color codes based on speaker to visually reinforce who is talking.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    width, height = 1280, 720
    
    # Theme colors
    if speaker.lower() == "expert":
        bg_color = (40, 55, 71) # Dark Blue-Grey for Expert
        accent_color = (100, 200, 255)
    else:
        bg_color = (200, 70, 70) # Muted Red/Maroon for Host? Or maybe just Dark Slate?
        # Let's go with a vibrant Dark Purple for Host
        bg_color = (60, 20, 60)
        accent_color = (255, 100, 200)

    # Gradient simulation (simple) - just solid for now to save complexity
    img = Image.new('RGB', (width, height), bg_color)
    d = ImageDraw.Draw(img)
    
    try:
        # Try a slightly better font if available
        font_large = ImageFont.truetype("arial.ttf", 70)
        font_small = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
    # Draw Speaker Tag
    d.text((50, 50), speaker.upper(), font=font_small, fill=accent_color)
    
    # Wrap text helper
    def draw_wrapped_text(text, font, y_start):
        lines = []
        words = text.split()
        current_line = []
        for word in words:
            # check width
            test_line = ' '.join(current_line + [word])
            bbox = d.textbbox((0, 0), test_line, font=font)
            if bbox[2] < 1100: # specific width limit
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        
        y = y_start
        for line in lines:
            bbox = d.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            x = (width - line_w) / 2
            d.text((x, y), line, font=font, fill=(255, 255, 255))
            y += bbox[3] - bbox[1] + 10
        return y

    # Draw Title (the visual prompt or main text summary)
    # If title is too long, we truncate or wrap
    y_pos = 250
    draw_wrapped_text(title, font_large, y_pos)
    
    if subtitle:
        # Show subtitle at bottom?
        pass
    
    filename = f"text_slide_{random.randint(1000,9999)}.png"
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
        
        # 2. Get Visual
        # Did we generate a specific diagram for this scene?
        visual_info = next((v for v in visual_assets if v["scene"] == i), None)
        image_path = None
        
        if visual_info and visual_info.get("local_path"):
            image_path = visual_info["local_path"]
        else:
            # Fallback: Create text slide based on visual_prompt or text summary
            # We use visual_prompt if it's text-friendly, otherwise just the speaker name/context
            slide_text = scene.get("visual_prompt", "")
            if len(slide_text) > 50 or "diagram" in slide_text.lower():
                # If prompt is too long/complex description, maybe just use the scene text summary?
                # Actually, let's just show what they are talking about (first 10 words of text)
                slide_text = " ".join(scene.get("text", "").split()[:8]) + "..."
            
            image_path = create_text_slide(slide_text, speaker=scene.get("speaker", "Host"))
            
        # 3. Create Clip
        try:
            # MoviePy v2 compatibility
            img_clip = ImageClip(image_path).with_duration(duration)
            img_clip = img_clip.with_audio(audio_clip)
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

def run_video_workflow(problem_text):
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
    
    # 2. Process Visuals (Diagrams)
    # Only generate diagrams if clearly requested in visual_prompt
    for i, scene in enumerate(script.get("scenes", [])):
        v_prompt = scene.get("visual_prompt", "").lower()
        has_diagram = "diagram" in v_prompt or "chart" in v_prompt or "graph" in v_prompt
        
        if has_diagram:
            print(f"Generatign Diagram for Scene {i+1}...")
            url = generate_kroki_diagram(scene["visual_prompt"])
            if url:
                local_path = download_image(url, os.path.join(MEDIA_ROOT, "images"))
                if local_path:
                    visual_asset = {"type": "image", "scene": i, "url": url, "local_path": local_path}
                    results["assets"].append(visual_asset)
                    visual_assets.append(visual_asset)
    
    # 3. Audio
    audio_files = []
    if HAS_GTTS:
        print("Generating Voiceovers...")
        audio_files = generate_audio_assets(script, os.path.join(MEDIA_ROOT, "audio"))
        results["audio_files"] = audio_files
    
    # 4. Assembly
    video_url = None
    if HAS_MOVIEPY and audio_files:
        print("Assembling video segments...")
        video_path = assemble_video(script, audio_files, visual_assets, os.path.join(MEDIA_ROOT, "video"))
        if video_path:
            # Convert absolute path to relative URL
            # Note: in Django settings.MEDIA_URL is usually '/media/'
            # We assume MEDIA_ROOT is mapped to /media/ URL.
            # Filename is the last part
            fname = os.path.basename(video_path)
            video_url = f"/media/video/{fname}"
            results["video_url"] = video_url
            print(f"Video created: {video_url}")
        else:
            results["video_error"] = "Assembly failed"
    else:
        results["video_error"] = "MoviePy missing or no audio"
        
    return results
