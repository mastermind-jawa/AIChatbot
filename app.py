from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import os
import base64
import io
from pypdf import PdfReader
from PIL import Image

load_dotenv()

app = Flask(__name__)

# Keep in-memory conversation history
SYSTEM_PERSONA = (
    "You are Sudharshini AI, a brilliant, friendly, intelligent allrounder AI assistant. "
    "You excel at answering questions, programming, creative writing, document analysis, "
    "and detailed visual understanding of diagrams, schematics, flowcharts, and photos. "
    "Provide clear, structured, well-formatted markdown responses."
)

conversation_history = [
    {"role": "system", "content": SYSTEM_PERSONA}
]

def extract_text_from_pdf(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n" + page_text
        return text.strip()
    except Exception as e:
        return f"[PDF Extraction Error: {str(e)}]"

def optimize_image(image_bytes):
    """Downscale and optimize image for fast visual inference."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            max_size = 1600
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, "image/jpeg"

def analyze_image_with_vision(image_bytes, mime_type, user_prompt, file_name="image"):
    """Dedicated high-fidelity multimodal vision engine for diagrams, flowcharts, circuits, and photos."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return "Image analysis error: GEMINI_API_KEY is not configured.", False

    client = genai.Client(api_key=gemini_key)
    vision_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-3.7-flash",
        "gemini-3.6-flash"
    ]

    prompt_text = user_prompt or (
        "Please analyze and explain this image thoroughly. "
        "If it is an architecture, circuit, or flowchart diagram, describe each section, component, "
        "data flow, and hardware/software connections in detail."
    )

    last_err = None
    for model in vision_models:
        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model=model,
                contents=[
                    image_part,
                    types.Part.from_text(text=f"System: {SYSTEM_PERSONA}\n\nUser Question: {prompt_text}")
                ]
            )
            if response and response.text:
                return response.text, True
        except Exception as e:
            last_err = e
            continue

    return f"Image vision processing error: {str(last_err)}", False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    file_b64 = (data.get("file") or data.get("image") or "").strip()
    file_name = data.get("file_name", "")
    user_api_key = (data.get("api_key") or "").strip()
    model_name = data.get("model", "")

    if not user_message and not file_b64:
        return jsonify({"reply": "Please type a message or upload an image/document."}), 400

    # Parse base64 file if attached
    file_bytes = None
    mime_type = "image/jpeg"
    is_pdf = False

    if file_b64:
        try:
            if "," in file_b64:
                header, encoded = file_b64.split(",", 1)
                if "application/pdf" in header or file_name.lower().endswith(".pdf"):
                    mime_type = "application/pdf"
                    is_pdf = True
                elif "image/" in header:
                    mime_type = header.split(";")[0].replace("data:", "")
                file_bytes = base64.b64decode(encoded)
            else:
                if file_name.lower().endswith(".pdf"):
                    mime_type = "application/pdf"
                    is_pdf = True
                file_bytes = base64.b64decode(file_b64)
        except Exception:
            file_bytes = None

    # CASE 1: Image Upload -> High-Fidelity Multimodal Vision Pipeline
    if file_bytes and not is_pdf:
        opt_bytes, opt_mime = optimize_image(file_bytes)
        display_prompt = user_message or "Explain and analyze this uploaded image."
        conversation_history.append({
            "role": "user",
            "content": f"[User uploaded image: {file_name or 'image'}]\n{display_prompt}"
        })

        reply, success = analyze_image_with_vision(opt_bytes, opt_mime, user_message, file_name)
        if success:
            conversation_history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply, "provider": "vision", "model": "gemini-vision"})
        else:
            # If vision failed, pop user turn and return error
            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history.pop()
            return jsonify({"reply": reply, "error": True, "code": "vision_error"}), 500

    # CASE 2: Text Chat & PDF Document Analysis -> Ultra-Fast Groq LPU Pipeline
    hist_text = user_message
    if not hist_text and is_pdf:
        hist_text = f"[User uploaded PDF: {file_name}]"
    conversation_history.append({"role": "user", "content": hist_text})

    groq_key = user_api_key if user_api_key else os.getenv("GROQ_API_KEY", "")

    # Groq LPU execution
    if groq_key:
        try:
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            valid_groq_models = [
                "openai/gpt-oss-120b",
                "groq/compound",
                "groq/compound-mini",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b"
            ]
            if not model_name or model_name not in valid_groq_models:
                target_model = "openai/gpt-oss-120b"
            else:
                target_model = model_name

            messages_payload = conversation_history.copy()

            if file_bytes and is_pdf:
                pdf_text = extract_text_from_pdf(file_bytes)
                prompt_text = user_message or "Summarize and analyze this PDF document in detail."
                full_prompt = f"PDF Document Content ({file_name}):\n{pdf_text}\n\nUser Request: {prompt_text}"
                messages_payload[-1] = {"role": "user", "content": full_prompt}

            response = client.chat.completions.create(
                model=target_model,
                messages=messages_payload,
            )
            reply = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply, "provider": "groq", "model": target_model})
        except Exception as groq_err:
            groq_err_str = str(groq_err)
            # Fallback to Gemini if rate limited or Groq issue
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            if gemini_key:
                try:
                    g_client = genai.Client(api_key=gemini_key)
                    g_response = g_client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=[f"System: {SYSTEM_PERSONA}\n\nUser: {user_message}"]
                    )
                    reply = g_response.text
                    conversation_history.append({"role": "assistant", "content": reply})
                    return jsonify({"reply": reply, "provider": "gemini-fallback"})
                except Exception:
                    pass

            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history.pop()

            if "rate_limit_exceeded" in groq_err_str or "429" in groq_err_str:
                reply = "⚠️ Rate limit reached. Please wait a few moments and click 🔄 Retry."
                return jsonify({"reply": reply, "error": True, "code": "rate_limit_exhausted"}), 429
            return jsonify({"reply": f"Groq Error: {groq_err_str}", "error": True, "code": "api_error"}), 500

    return jsonify({"reply": "API key configuration missing. Please check .env file.", "error": True}), 400

@app.route("/reset", methods=["POST"])
def reset():
    """Clear conversation history."""
    global conversation_history
    conversation_history = [
        {"role": "system", "content": SYSTEM_PERSONA}
    ]
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
