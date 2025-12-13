# 📽️ Automated Workflow: Question Explainer Video (Video A)

**Goal:** To automatically generate a short video that presents an educational problem, the core concept, and a strategic hint (`Video A`), prompting the user to pause and attempt the solution.

**Key Principle:** Progressive Disclosure — The Answer (`Video B`) is deliberately withheld for a later video.

## ⚙️ Summary of API Stack (Free Tier Focus)

| Stage | Primary Function | Key Tool/API Used |
| :--- | :--- | :--- |
| **I. Scripting** | Content Structuring & Narration | **Gemini API** (LLM) + **Google Cloud TTS** (Audio) |
| **II. Asset Creation** | Generating Diagrams & B-Roll | **Specialized AI Solvers** / **Adobe Firefly** (Visuals) |
| **III. Assembly** | Syncing Audio and Visuals | **Creatomate / Shotstack API** (Programmatic Video Editing) |

---

## I. Stage 1: The Scripting Engine (Content Generation)

This stage is handled by your server-side code (e.g., Python/Node.js) orchestrating the LLM to produce structured content.

| Step | Action | Tool/API | Output Data | Description |
| :--- | :--- | :--- | :--- | :--- |
| **1.1 Input Problem** | System receives the user's educational problem text. | **Your Backend Code** | Text Input | The workflow trigger. |
| **1.2 Generate Structured Script** | **Critical Step:** Prompt the Gemini API to output a 3-part script (Question, Concept, Hint) as a structured **JSON** object, including `narration_text` and a `visual_prompt` for each scene. | **Gemini API** (Free Tier) | **`Script_A_JSON`** | Guarantees machine-readable, segment-by-segment instructions for the video builder. |
| **1.3 Extract Narration Text** | Concatenate all `narration_text` fields into a single string. | **Your Backend Code** | Plain Text String | The complete script, ready for audio generation. |
| **1.4 Generate Voiceover** | Convert the final narration string into a high-quality audio file. | **Google Cloud TTS API** (Free Tier) | **`Audio_A.mp3`** | The synchronized audio track that dictates the video's timing. |

### 🔑 Key Output from Stage 1 (`Script_A_JSON` Example)

The JSON must explicitly define scene duration and visual action:

```json
{
  "scenes": [
    {
      "duration": 5.0,
      "narration_text": "Welcome! Today we tackle the classic problem of the 2kg block on a frictionless ramp.",
      "visual_prompt": "Display the full question text over a clean, animated background."
    },
    {
      "duration": 10.0,
      "narration_text": "To solve this, the core concept you must apply is the **Conservation of Mechanical Energy**.",
      "visual_prompt": "Abstract animation of energy bars shifting, emphasizing 'Conservation of Energy'."
    },
    {
      "duration": 8.0,
      "narration_text": "Your key hint: start by defining your initial and final states and drawing your forces. Pause now and set up your variables!",
      "visual_prompt": "Display a generated Free-Body Diagram (FBD) prominently with a 'PAUSE HERE' call-to-action."
    }
  ]
}

---

## II. Stage 2: Visual Asset Generation

This stage programmatically creates the specific, custom visual elements needed for the explanation, guided by the `visual_prompt` metadata generated in Stage 1.

| Step | Action | Tool/API | Output File | Description |
| :--- | :--- | :--- | :--- | :--- |
| **2.1 Create Technical Diagram (The Hint)** | Based on the question and the AI's `visual_prompt` (e.g., "Draw a Free-Body Diagram" or "Generate a flowchart"), create the precise technical graphic. | **Kroki.io** (Free Service) or **Eraser DiagramGPT** (Free Tier/Presets) | **`FBD_Diagram.png`** | **Critical:** Kroki.io is an excellent free service that converts code-based text (like PlantUML, Mermaid) into high-quality diagrams (SVG/PNG). This provides a reliable API endpoint for creating accurate educational visuals. |
| **2.2 Generate Background B-Roll** | Based on the abstract prompts (e.g., "dynamic energy transfer"), generate a short visual clip or image. | **Adobe Firefly** (Free Credits) or **Canva Magic Media** (Free) | **`Concept_Background.mp4`** and **`Pause_Screen.png`** | Provides visual context for the **Core Concept** and the final **Call-to-Action** screen. |
| **2.3 Upload Assets** | Upload all generated files (`.png`, `.mp4`, `.mp3` from Stage 1) to a public-facing cloud storage. | **Cloudinary / AWS S3** (Free Tiers) | **Public URLs** | Video assembly APIs require public URLs to fetch and include media files in the final render. |

## III. Stage 3: Automated Video Assembly and Delivery

This final stage uses a programmatic video editing API to stitch the audio and visual assets together according to the script's timing, producing the final MP4 file.

| Step | Action | Tool/API | Output | Description |
| :--- | :--- | :--- | :--- | :--- |
| **3.1 Define Video Template** | **Manual/One-Time:** Design the reusable layout (logo, text style) in the platform's editor. Crucially, mark elements as **dynamic fields** (e.g., `[Title Text]`, `[Hint Image Placeholder]`). | **Creatomate API / Shotstack API** (Trial/Free Keys) | **`Template_ID_A`** | Establishes the video's visual brand identity structure. |
| **3.2 Map & Structure Render Call** | **Core Automation:** Your code constructs the final JSON payload. This payload references the `Template_ID_A` and inserts the data: the full `Script_A_JSON` structure, the URL of `Audio_A.mp3`, and the public URL of `FBD_Diagram.png`. | **Your Backend Code** | Final JSON Payload | This is the command that tells the API **what** to show and **when** to show it (e.g., start displaying `FBD_Diagram.png` at 0:15). |
| **3.3 Programmatic Render Call** | Send the structured JSON payload to the assembly service. | **Creatomate API / Shotstack API** (Trial/Free Keys) | **API Request Sent.** | The chosen platform handles the complex, server-side rendering process in the cloud. |
| **3.4 Final Retrieval and Host** | The system monitors the API for the "render complete" status, receives the final video URL (often via a webhook), and stores it. | **Your Hosting/CDN** | **Final Output: `Video_A.mp4`** | The complete Question Explainer Video file, ready for presentation. |

---

### Synchronization Key

The success of the final video hinges on accurately timing the visual assets. The JSON payload sent in **Step 3.2** must use the scene durations provided by the **Gemini API** (Stage 1) to instruct the Video Assembly API exactly when to reveal the `FBD_Diagram.png` and transition to the final "Pause" screen.