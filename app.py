from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import os
import sys
import base64
import io
import time
from datetime import datetime, timezone
from pypdf import PdfReader
from PIL import Image
import pymongo
from pymongo import MongoClient

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ----------------- MongoDB Database Setup -----------------
MONGODB_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI") or ""
DB_NAME = os.getenv("MONGODB_DB_NAME", "sudharshini_ai")

mongo_client = None
db = None
messages_collection = None

def get_mongodb():
    """Lazily initializes and returns MongoDB database handle with timeout safeguards."""
    global mongo_client, db, messages_collection
    if not MONGODB_URI:
        return None, None
    if messages_collection is not None:
        return db, messages_collection
    try:
        mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000,
            retryWrites=True
        )
        mongo_client.admin.command("ping")
        db = mongo_client[DB_NAME]
        messages_collection = db["chat_messages"]
        try:
            messages_collection.create_index([("session_id", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)])
        except Exception:
            pass
        return db, messages_collection
    except Exception as e:
        print(f"[MongoDB Notice] {e}")
        return None, None

def save_message_to_db(session_id, role, content, meta=None):
    _, col = get_mongodb()
    if col is not None:
        try:
            doc = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc),
                "meta": meta or {}
            }
            col.insert_one(doc)
            return True
        except Exception as e:
            print(f"[MongoDB Save Error]: {e}")
    return False

def get_session_history_from_db(session_id, limit=40):
    _, col = get_mongodb()
    if col is not None:
        try:
            cursor = col.find({"session_id": session_id}).sort("timestamp", pymongo.ASCENDING).limit(limit)
            history = []
            for doc in cursor:
                history.append({
                    "role": doc.get("role", "user"),
                    "content": doc.get("content", ""),
                    "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None,
                    "meta": doc.get("meta", {})
                })
            return history
        except Exception as e:
            print(f"[MongoDB Query Error]: {e}")
    return None

def clear_session_history_in_db(session_id):
    _, col = get_mongodb()
    if col is not None:
        try:
            col.delete_many({"session_id": session_id})
            return True
        except Exception as e:
            print(f"[MongoDB Delete Error]: {e}")
    return False

# In-memory fallback
in_memory_sessions = {}

SYSTEM_PERSONA = (
    "You are Sudharshini AI, a brilliant, friendly, intelligent allrounder AI assistant. "
    "You excel at answering questions, programming, creative writing, document analysis, "
    "and detailed visual understanding of diagrams, schematics, flowcharts, and photos. "
    "Provide clear, structured, well-formatted markdown responses."
)

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
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        return "Image analysis error: GEMINI_API_KEY is not configured.", False

    try:
        client = genai.Client(api_key=gemini_key)
    except Exception as e:
        return f"Gemini client initialization failed: {str(e)}", False

    vision_models = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-flash-latest"]

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

def run_gemini_fallback(user_message, messages_history=None):
    """Executes Gemini fallback inference with multi-model failover."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        return None
    candidate_models = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-flash-latest", "gemini-3.5-flash-lite"]
    try:
        client = genai.Client(api_key=gemini_key)
        contents = [f"System: {SYSTEM_PERSONA}"]
        if messages_history:
            for m in messages_history[-8:]:
                if m.get("role") in ("user", "assistant"):
                    prefix = "User: " if m["role"] == "user" else "Assistant: "
                    contents.append(prefix + m["content"])
        contents.append(f"User: {user_message}")
        
        for m in candidate_models:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=contents
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                print(f"[Gemini Model {m} failed]: {e}")
                continue
    except Exception as e:
        print(f"[Gemini Fallback Error]: {e}")
    return None

# ----------------- Route Handlers -----------------
def handle_home():
    return render_template("index.html")

def handle_status():
    _, col = get_mongodb()
    mongo_connected = col is not None
    return jsonify({
        "status": "online",
        "mongodb_connected": mongo_connected,
        "database": DB_NAME if mongo_connected else None,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    })

def handle_history():
    session_id = request.args.get("session_id", "default_session").strip()
    db_history = get_session_history_from_db(session_id)
    if db_history is not None:
        return jsonify({"history": db_history, "source": "mongodb"})
    
    mem_hist = in_memory_sessions.get(session_id, [])
    filtered = [m for m in mem_hist if m.get("role") in ("user", "assistant")]
    return jsonify({"history": filtered, "source": "memory"})

def handle_reset():
    data = request.json or {}
    session_id = (data.get("session_id") or "default_session").strip()
    clear_session_history_in_db(session_id)
    in_memory_sessions[session_id] = [
        {"role": "system", "content": SYSTEM_PERSONA}
    ]
    return jsonify({"status": "reset", "session_id": session_id})

def handle_chat():
    data = request.json or {}
    session_id = (data.get("session_id") or "default_session").strip()
    user_message = data.get("message", "").strip()
    file_b64 = (data.get("file") or data.get("image") or "").strip()
    file_name = data.get("file_name", "")
    user_api_key = (data.get("api_key") or "").strip()
    model_name = data.get("model", "")

    if not user_message and not file_b64:
        return jsonify({"reply": "Please type a message or upload an image/document."}), 400

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

    db_history = get_session_history_from_db(session_id)
    if db_history is not None:
        formatted_history = [{"role": "system", "content": SYSTEM_PERSONA}]
        for item in db_history[-10:]:
            formatted_history.append({"role": item["role"], "content": item["content"]})
    else:
        if session_id not in in_memory_sessions:
            in_memory_sessions[session_id] = [{"role": "system", "content": SYSTEM_PERSONA}]
        formatted_history = in_memory_sessions[session_id].copy()

    # Image Analysis
    if file_bytes and not is_pdf:
        opt_bytes, opt_mime = optimize_image(file_bytes)
        display_prompt = user_message or "Explain and analyze this uploaded image."
        user_entry = f"[User uploaded image: {file_name or 'image'}]\n{display_prompt}"
        
        reply, success = analyze_image_with_vision(opt_bytes, opt_mime, user_message, file_name)
        if success:
            save_message_to_db(session_id, "user", user_entry, {"file_name": file_name, "type": "image"})
            save_message_to_db(session_id, "assistant", reply, {"provider": "gemini-vision"})
            if session_id in in_memory_sessions:
                in_memory_sessions[session_id].append({"role": "user", "content": user_entry})
                in_memory_sessions[session_id].append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply, "provider": "vision", "model": "gemini-vision"})
        else:
            return jsonify({"reply": reply, "error": True, "code": "vision_error"}), 500

    # Text & PDF Chat
    hist_text = user_message
    if not hist_text and is_pdf:
        hist_text = f"[User uploaded PDF: {file_name}]"

    groq_key = user_api_key if user_api_key else os.getenv("GROQ_API_KEY", "").strip()

    # Try Groq if key is present
    if groq_key and len(groq_key) > 10:
        try:
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            valid_groq_models = [
                "openai/gpt-oss-120b",
                "groq/compound",
                "groq/compound-mini",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b"
            ]
            target_model = model_name if model_name in valid_groq_models else "openai/gpt-oss-120b"

            messages_payload = formatted_history.copy()
            if file_bytes and is_pdf:
                pdf_text = extract_text_from_pdf(file_bytes)
                prompt_text = user_message or "Summarize and analyze this PDF document in detail."
                full_prompt = f"PDF Document Content ({file_name}):\n{pdf_text}\n\nUser Request: {prompt_text}"
                messages_payload.append({"role": "user", "content": full_prompt})
            else:
                messages_payload.append({"role": "user", "content": hist_text})

            response = client.chat.completions.create(
                model=target_model,
                messages=messages_payload,
            )
            reply = response.choices[0].message.content
            save_message_to_db(session_id, "user", hist_text, {"file_name": file_name if is_pdf else None})
            save_message_to_db(session_id, "assistant", reply, {"provider": "groq", "model": target_model})
            if session_id in in_memory_sessions:
                in_memory_sessions[session_id].append({"role": "user", "content": hist_text})
                in_memory_sessions[session_id].append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply, "provider": "groq", "model": target_model})
        except Exception as groq_err:
            print(f"[Groq Error, switching to Gemini]: {groq_err}")

    # Fallback to Gemini 3.6 Flash
    gemini_reply = run_gemini_fallback(hist_text, formatted_history)
    if gemini_reply:
        save_message_to_db(session_id, "user", hist_text)
        save_message_to_db(session_id, "assistant", gemini_reply, {"provider": "gemini-3.6-flash"})
        if session_id in in_memory_sessions:
            in_memory_sessions[session_id].append({"role": "user", "content": hist_text})
            in_memory_sessions[session_id].append({"role": "assistant", "content": gemini_reply})
        return jsonify({"reply": gemini_reply, "provider": "gemini", "model": "gemini-3.6-flash"})

    return jsonify({
        "reply": "Could not connect to AI services. Please verify your GEMINI_API_KEY or GROQ_API_KEY in Vercel Environment Variables.",
        "error": True
    }), 500

# ----------------- Unified Catch-All Router -----------------
@app.route("/", defaults={"subpath": ""}, methods=["GET", "POST", "OPTIONS"])
@app.route("/<path:subpath>", methods=["GET", "POST", "OPTIONS"])
def unified_router(subpath=""):
    # Strip any trailing slashes or query parameters
    p = subpath.lower().strip("/").split("?")[0]
    
    if p in ("api/status", "status"):
        return handle_status()
    elif p in ("api/history", "history"):
        return handle_history()
    elif p in ("api/reset", "reset"):
        return handle_reset()
    elif p in ("api/chat", "chat") or request.method == "POST":
        return handle_chat()
    else:
        return handle_home()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
