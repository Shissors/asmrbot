"""
server.py
=========
FastAPI web server. Run it with:

    uvicorn server:app --reload --port 8000

Endpoints:
  GET  /               → serves static/index.html  (the UI)
  GET  /characters     → list of available characters
  POST /generate       → generates script + queues audio generation
  GET  /status/{id}    → poll for audio-ready status
  GET  /audio/{id}     → download the finished WAV file
"""

import os
import uuid
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load GEMINI_API_KEY from .env file
load_dotenv()

from generate import (
    generate_script,
    parse_script,
    load_omnivoice,
    generate_audio,
    CHARACTERS,
)

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="ASMR Roleplay Platform")

# Allow the browser to call the API (needed if you ever host frontend separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve everything in /static as public files (our index.html lives here)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Make sure the outputs folder exists
Path("outputs").mkdir(exist_ok=True)

# ── Global state ───────────────────────────────────────────────────────────
tts_model = None          # OmniVoice model (loaded once at startup)
executor  = ThreadPoolExecutor(max_workers=1)   # One TTS job at a time

# Dict that stores every generation job
# key   = session_id (UUID)
# value = { status, script, audio_path, error }
jobs: dict = {}


# ── Startup: load TTS model ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global tts_model
    tts_model = load_omnivoice()   # blocks until model is ready (~30 s)


# ── Request model ──────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    topic     : str
    mood      : str   = "calming and soothing"
    duration  : int   = 120   # seconds
    character : str   = "yae_miko"
    speed     : float = 0.88


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def home():
    """Serve the main page."""
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/characters")
async def list_characters():
    """Return available characters so the UI can populate a dropdown."""
    return [{"id": k, "name": v["name"]} for k, v in CHARACTERS.items()]


@app.post("/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Two-phase generation:
      Phase 1 (fast, ~2 s):  Gemini writes the tagged ASMR script.
                              We return it immediately so the user can read it.
      Phase 2 (slow, varies): OmniVoice generates audio in the background.
                              Client polls /status/{id} until "ready".
    """
    # ── Validate inputs ───────────────────────────────────────────────
    if req.character not in CHARACTERS:
        return JSONResponse({"error": f"Unknown character '{req.character}'"}, 400)
    if not (10 <= req.duration <= 600):
        return JSONResponse({"error": "Duration must be 10–600 seconds."}, 400)

    # ── Phase 1: generate script ──────────────────────────────────────
    print(f"\n📝  Generating script: '{req.topic}' ({req.duration}s)")
    script = generate_script(req.topic, req.mood, req.duration, req.character)

    # ── Create a job record ───────────────────────────────────────────
    session_id = str(uuid.uuid4())
    audio_path = f"outputs/{session_id}.wav"

    jobs[session_id] = {
        "status"     : "generating",
        "script"     : script,
        "audio_path" : audio_path,
    }

    # ── Phase 2: audio in the background ─────────────────────────────
    background_tasks.add_task(
        _background_audio,
        session_id, script, req.character, audio_path, req.speed
    )

    return {
        "session_id" : session_id,
        "script"     : script,
        "status"     : "generating",
    }


async def _background_audio(session_id, script, character, audio_path, speed):
    """Runs OmniVoice in a thread so it doesn't block the web server."""
    loop = asyncio.get_event_loop()
    try:
        segments = parse_script(script)
        print(f"🎙️   [{session_id[:8]}]  {len(segments)} segments to generate…")

        # run_in_executor lets blocking (CPU/GPU) code run without freezing FastAPI
        await loop.run_in_executor(
            executor,
            lambda: generate_audio(tts_model, segments, character, audio_path, speed)
        )
        jobs[session_id]["status"] = "ready"
        print(f"✅  [{session_id[:8]}]  Audio ready.")

    except Exception as exc:
        jobs[session_id]["status"] = "error"
        jobs[session_id]["error"]  = str(exc)
        print(f"❌  [{session_id[:8]}]  Error: {exc}")


@app.get("/status/{session_id}")
async def status(session_id: str):
    """
    Poll this endpoint to check if audio is ready.
    Returns: { "status": "generating" | "ready" | "error", "error": str | null }
    """
    if session_id not in jobs:
        return JSONResponse({"error": "Session not found."}, 404)
    job = jobs[session_id]
    return {"status": job["status"], "error": job.get("error")}


@app.get("/audio/{session_id}")
async def get_audio(session_id: str):
    """Download the finished WAV once status == 'ready'."""
    if session_id not in jobs:
        return JSONResponse({"error": "Session not found."}, 404)
    job = jobs[session_id]
    if job["status"] != "ready":
        return JSONResponse({"error": "Audio not ready yet."}, 425)
    return FileResponse(
        job["audio_path"],
        media_type="audio/wav",
        filename="asmr_roleplay.wav",
    )