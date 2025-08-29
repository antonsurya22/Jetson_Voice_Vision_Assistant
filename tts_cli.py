#!/usr/bin/env python3
import sys, json, subprocess
from pathlib import Path

# Use the venv's Piper module (installed as "piper-tts")
PIPER_CMD = [sys.executable, "-m", "piper"]
VOICES_DIR = Path(__file__).parent / "voices"

VOICE_MAP = {
    "en": "en_US-ryan",     # expects en_US-ryan.onnx + en_US-ryan.onnx.json in voices/
    "zh": "zh_CN-huayan",   # expects zh_CN-huayan.onnx + zh_CN-huayan.onnx.json
}
DEFAULT_VOICE = "en_US-ryan"

def speak(text, lang):
    key = lang if lang in VOICE_MAP else ("zh" if lang.startswith("zh") and "zh" in VOICE_MAP else "en")
    voice = VOICE_MAP.get(key, DEFAULT_VOICE)

    onnx = VOICES_DIR / f"{voice}.onnx"
    cfg  = VOICES_DIR / f"{voice}.onnx.json"  # Piper will auto-load this if present

    # Fallback to default if missing
    if not onnx.exists() or not cfg.exists():
        voice = DEFAULT_VOICE
        onnx = VOICES_DIR / f"{voice}.onnx"
        cfg  = VOICES_DIR / f"{voice}.onnx.json"

    tmp_wav = "/tmp/piper_tts.wav"
    try:
        # DO NOT pass --config; Piper finds <model>.onnx.json automatically
        subprocess.run(
            PIPER_CMD + ["--model", str(onnx), "--output_file", tmp_wav],
            input=text.encode("utf-8"),
            check=True
        )
        subprocess.run(["aplay", "-q", tmp_wav], check=False)
    except subprocess.CalledProcessError as e:
        print("[TTS] Piper failed:", e, file=sys.stderr)

def main():
    if sys.stdin.isatty():
        print('Usage: echo \'{"text":"你好","lang":"zh"}\' | python3 tts_cli.py', file=sys.stderr)
        sys.exit(1)
    data = json.loads(sys.stdin.read())
    text = data.get("text","").strip()
    lang = data.get("lang","en")
    if text:
        speak(text, lang)

if __name__ == "__main__":
    main()
