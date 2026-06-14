"""
generate.py
===========
The heart of the ASMR platform. Two jobs:

  1. generate_script()  → asks Gemini to write a tagged ASMR script
  2. generate_audio()   → uses OmniVoice to turn that script into a WAV file

Data flow:
  user input
    → generate_script()  →  raw script with emotion tags
    → parse_script()     →  list of segments (speech / sound / pause)
    → generate_audio()   →  .wav file
"""

import re
import os
import numpy as np
import soundfile as sf
import torch
from google import genai # UPDATED IMPORT


# ── Gemini setup ──────────────────────────────────────────────────────────
# Uses the FREE tier: Gemini 1.5 Flash (1,500 requests/day, no credit card)
# Get your key at https://aistudio.google.com/

# UPDATED CLIENT INITIALIZATION
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# ══════════════════════════════════════════════════════════════════════════
# CHARACTER PROFILES
# Add new characters here when you want to expand the platform.
# ══════════════════════════════════════════════════════════════════════════

CHARACTERS = {
    "yae_miko": {
        "name": "Yae Miko",

        # Path to a 5-10 second clean voice sample of this character.
        # OmniVoice uses this to clone the voice.
        "ref_audio": "voices/yae_miko_ref.wav",
        "ref_text":  "I am the Guuji of the Grand Narukami Shrine. The purpose of my visit is to monitor your every move, for such is the order of the shrine.",

        # Injected into the Gemini prompt to shape the script's tone.
        "personality": """\
You are Yae Miko from Genshin Impact — the Grand Naganohara Shrine Maiden.
You are elegant, mischievous, and faintly condescending, yet secretly warm.
You speak in a teasing, melodic cadence. You call the listener "little mortal"
or "dear visitor". You occasionally make sly fox references.
For ASMR, you become intimate and hushed, sharing secrets only the listener may hear.\
""",

        # OmniVoice Voice-Design string used for [whisper] sections.
        # Voice Design doesn't need a reference audio file.
        "whisper_instruct": "female, high pitch, whisper",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# ASMR TAG SYSTEM
# These tags are injected into the Gemini prompt.
# The script parser (parse_script) turns them into OmniVoice calls.
# ══════════════════════════════════════════════════════════════════════════

TAG_GUIDE = """
You must embed these tags throughout the script to control voice and pacing:

  [whisper]text[/whisper]  → whisper this part (breathy, intimate)
  [soft]text[/soft]        → slightly quieter and slower
  [pause:N]                → N seconds of silence  e.g. [pause:2]
  [chuckle]                → a short soft laugh
  [sigh]                   → a gentle sigh
  [breathe]                → a slow audible in-breath
  [hm]                     → a thoughtful hum

RULES:
• Use at least one tag per paragraph.
• Put [pause] after important lines — let the words breathe.
• Use [breathe] before emotional sentences for ASMR effect.
• Whisper mode is perfect for intimate revelations or encouragement.
"""


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — SCRIPT GENERATION  (Gemini API)
# ══════════════════════════════════════════════════════════════════════════

def generate_script(topic: str, mood: str, duration_seconds: int,
                    character: str = "yae_miko") -> str:
    """
    Calls Gemini 3.5 Flash (free tier) to write a tagged ASMR script.
    """
    char = CHARACTERS[character]

    # At ~90 words per minute (ASMR pace), estimate how many words we need.
    target_words = int(duration_seconds / 60 * 90)

    prompt = f"""You are {char['name']} performing an ASMR roleplay for a listener
who wants to relax and feel immersed.

CHARACTER PROFILE:
{char['personality']}

YOUR TASK:
Write an ASMR script for this scenario:
  • Topic    : {topic}
  • Mood     : {mood}
  • Length   : approximately {target_words} words (~{duration_seconds} seconds at ASMR pace)

{TAG_GUIDE}

OUTPUT RULES:
- Speak directly to "you" (the listener) at all times.
- Include rich sensory details: sounds, textures, warmth, scents.
- Stay fully in character the entire time.
- Output ONLY the script — no titles, no stage directions, no commentary.
- Begin speaking immediately with your first sentence."""

    # UPDATED GENERATION SYNTAX
    response = client.models.generate_content(
        model="gemini-3.5-flash", # Fixed the model name from 3.5 to 1.5
        contents=prompt
    )
    return response.text.strip()


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — SCRIPT PARSING
# ══════════════════════════════════════════════════════════════════════════

def parse_script(script: str) -> list:
    # Map our custom tags → OmniVoice built-in non-verbal tags
    SOUND_MAP = {
        "chuckle": "[laughter]",
        "breathe": "[sigh]",
        "sigh":    "[sigh]",
        "hm":      "[confirmation-en]",
    }

    segments    = []
    current_mode = "normal"   # tracks whether we're inside [whisper] or [soft]
    text_buffer  = ""         # accumulates text between tags

    def flush():
        nonlocal text_buffer
        text = text_buffer.strip()
        if text:
            segments.append({"type": "speech", "text": text, "mode": current_mode})
        text_buffer = ""

    TAG_RE = re.compile(r'\[(/?)(\w+)(?::([0-9.]+))?\]')
    last_pos = 0

    for m in TAG_RE.finditer(script):
        text_buffer += script[last_pos:m.start()]
        last_pos = m.end()

        closing = m.group(1) == "/"
        tag     = m.group(2).lower()
        value   = m.group(3)

        if tag == "pause":
            flush()
            segments.append({"type": "pause", "seconds": float(value or 1)})

        elif tag in SOUND_MAP and not closing:
            flush()
            segments.append({"type": "sound", "tag": SOUND_MAP[tag]})

        elif tag in ("whisper", "soft") and not closing:
            flush()
            current_mode = tag

        elif closing and tag in ("whisper", "soft"):
            flush()
            current_mode = "normal"

    text_buffer += script[last_pos:]
    flush()

    return segments


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — AUDIO GENERATION  (OmniVoice)
# ══════════════════════════════════════════════════════════════════════════

def load_omnivoice():
    from omnivoice import OmniVoice

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"⏳  Loading OmniVoice on {device}  (first run downloads ~2 GB)…")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    print("✅  OmniVoice ready!")
    return model


def generate_audio(model, segments: list, character: str,
                   output_path: str, speed: float = 0.88) -> str:
    char        = CHARACTERS[character]
    SAMPLE_RATE = 24_000   # OmniVoice always outputs at 24 kHz
    audio_parts = []

    for i, seg in enumerate(segments):
        kind = seg["type"]
        print(f"  [{i+1}/{len(segments)}] {kind}" +
              (f" [{seg.get('mode')}]" if kind == "speech" else ""))

        if kind == "pause":
            silence = np.zeros(int(seg["seconds"] * SAMPLE_RATE), dtype=np.float32)
            audio_parts.append(silence)

        elif kind == "sound":
            audio = model.generate(
                text      = seg["tag"],
                ref_audio = char["ref_audio"],
                ref_text  = char["ref_text"],
            )
            audio_parts.append(audio[0].astype(np.float32))

        elif kind == "speech":
            mode = seg["mode"]

            if mode == "whisper":
                audio = model.generate(
                    text    = seg["text"],
                    instruct = char["whisper_instruct"],
                    speed   = speed * 0.9,
                )
            else:
                actual_speed = speed * (0.85 if mode == "soft" else 1.0)
                audio = model.generate(
                    text      = seg["text"],
                    ref_audio = char["ref_audio"],
                    ref_text  = char["ref_text"],
                    speed     = actual_speed,
                )

            audio_parts.append(audio[0].astype(np.float32))

    if not audio_parts:
        raise ValueError("No audio segments were generated.")

    gap   = np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float32)
    final = audio_parts[0]
    for part in audio_parts[1:]:
        final = np.concatenate([final, gap, part])

    peak = np.abs(final).max()
    if peak > 0:
        final = final / peak * 0.85

    sf.write(output_path, final, SAMPLE_RATE, subtype="PCM_16")
    duration_sec = len(final) / SAMPLE_RATE
    print(f"✅  Saved {output_path}  ({duration_sec:.1f} s)")
    return output_path