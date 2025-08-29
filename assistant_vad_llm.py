#!/usr/bin/env python3
import sys, os, time, json, re, queue, subprocess, requests
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# ---------- Config (tuned for responsiveness) ----------
SAMPLE_RATE     = 16000
FRAME_MS        = 30           # 10/20/30 allowed; 30ms gives steadier VAD
VAD_AGGR        = 2            # 0..3 (2 is a good balance)
SILENCE_END_MS  = 400          # stop soon after silence
MIN_UTTER_MS    = 280          # ignore micro blips
MAX_UTTER_SEC   = 8

# Whisper settings (fast). Change to "base" if you want a tad more accuracy.
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "tiny")
DEVICE      = "cpu"
COMPUTE     = "int8"
MODEL_DIR   = Path(__file__).parent / "models" / "whisper"

# Ollama local LLM
OLLAMA_URL   = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct")

DET_PATH = "/tmp/detections.json"

# Choose mic:
#   - Set MIC_INDEX env:  export MIC_INDEX=2
#   - or we auto-pick a sensible input device.
MIC_INDEX_ENV = os.getenv("MIC_INDEX")

# ---------- Helpers ----------
def write_wav_pcm16(path, audio_f32, sr=SAMPLE_RATE):
    import wave
    if audio_f32.ndim > 1:
        audio_f32 = audio_f32[:, 0]
    a = np.clip(audio_f32, -1.0, 1.0)
    a = (a * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(a.tobytes())

def tts_say(text, lang):
    payload = json.dumps({"text": text, "lang": ("zh" if str(lang).startswith("zh") else "en")}, ensure_ascii=False)
    subprocess.run([sys.executable, "tts_cli.py"], input=payload, text=True)

def read_dets():
    try:
        with open(DET_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"ts":0,"items":[]}

def det_labels_string(max_items=12):
    det = read_dets()
    labels = []
    for d in det.get("items", []):
        lbl = d.get("label","")
        if lbl and lbl not in labels:
            labels.append(lbl)
        if len(labels) >= max_items: break
    return ", ".join(labels) if labels else ""

# vision intent (EN + 中文)
EN_VISION = re.compile(r"\b(what\s+do\s+you\s+see|what\s+are\s+you\s+seeing|look\s+around|describe|what'?s\s+in\s+front|what\s+is\s+on\s+the\s+camera)\b", re.I)
ZH_VISION = re.compile(r"(你看(到)?了?什麼|你看到什么|描述(一下)?|相機|摄像头|鏡頭|镜头)", re.I)

def is_vision_intent(text, lang):
    is_zh = str(lang).startswith("zh") or re.search(r"[\u4e00-\u9fff]", text)
    return (EN_VISION.search(text) and not is_zh) or (ZH_VISION.search(text) and is_zh)

def build_prompt(user_text, lang):
    sys_en = ("You are a concise voice assistant. "
              "Respond in the user's language. Keep answers short and spoken-friendly.")
    sys_zh = "你是一個簡潔的語音助理。請用用戶的語言回答，語氣自然口語化，回答要簡短。"
    system = sys_zh if (str(lang).startswith("zh") or re.search(r"[\u4e00-\u9fff]", user_text)) else sys_en

    if is_vision_intent(user_text, lang):
        labels = det_labels_string()
        if not labels:
            ctx_en = "Camera currently sees nothing notable."
            ctx_zh = "攝像頭目前沒有明顯物體。"
        else:
            ctx_en = f"Camera objects: {labels}."
            ctx_zh = f"攝像頭物體：{labels}。"
        context = ctx_zh if (str(lang).startswith("zh") or re.search(r"[\u4e00-\u9fff]", user_text)) else ctx_en
        user_text = f"{user_text}\n\n{context}\nAnswer based only on those objects."
    return system, user_text

def call_ollama(system, prompt, max_tokens=90):
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"<<SYS>>\n{system}\n<</SYS>>\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.9,
                "min_p": 0.05,
                "num_predict": max_tokens
            }
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response","").strip()
    except Exception as e:
        print("[LLM] Error:", e)
        return ""

# ---------- Streaming ASR with VAD + level meter ----------
class VADListener:
    def __init__(self, sr=SAMPLE_RATE, frame_ms=FRAME_MS, aggressiveness=VAD_AGGR):
        self.sr = sr
        self.frame_samples = int(sr * frame_ms / 1000)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.q = queue.Queue()
        self.last_rms = 0.0

    def _cb(self, indata, frames, tinfo, status):
        # indata: float32 [-1,1]
        audio = np.clip(indata[:,0], -1.0, 1.0)
        rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
        self.last_rms = rms
        pkt = (audio * 32767.0).astype(np.int16).tobytes()
        self.q.put(pkt)

    def stream(self, device=None):
        return sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                              blocksize=self.frame_samples, callback=self._cb,
                              device=device)

def pick_input_device():
    if MIC_INDEX_ENV is not None:
        try:
            idx = int(MIC_INDEX_ENV)
            sd.default.device = (None, idx)
            print(f"[Audio] Using MIC_INDEX={idx}")
            return idx
        except Exception as e:
            print(f"[Audio] MIC_INDEX invalid: {MIC_INDEX_ENV} ({e})")

    # Auto-pick: prefer names containing these keywords
    keywords = ["array", "usb", "mini", "mic", "respeaker", "seeed", "logitech", "c270"]
    try:
        devs = sd.query_devices()
        candidates = [i for i,d in enumerate(devs) if d.get("max_input_channels",0) > 0]
        preferred = [i for i in candidates if any(k in devs[i]["name"].lower() for k in keywords)]
        idx = (preferred or candidates)[0]
        sd.default.device = (None, idx)
        print(f"[Audio] Auto-selected input device {idx}: {devs[idx]['name']}")
        return idx
    except Exception as e:
        print("[Audio] Could not auto-select device:", e)
        return None

def main():
    # 1) Pick mic
    pick_input_device()

    # 2) Load Whisper once (download to local folder so cache is stable)
    print(f"[ASR] Loading Whisper ({WHISPER_SIZE}, {DEVICE}/{COMPUTE}) ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE,
                             download_root=str(MODEL_DIR))
    except Exception as e:
        print("[ASR] Model load failed, clearing cache and retrying…", e)
        import shutil, glob
        for p in glob.glob(str(MODEL_DIR / f"faster-whisper-{WHISPER_SIZE}*")):
            shutil.rmtree(p, ignore_errors=True)
        model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE,
                             download_root=str(MODEL_DIR))

    listener = VADListener()
    print("[Assistant] Live chat on. Speak anytime. Ctrl+C to stop.")
    last_lang = "en"

    # open stream
    with listener.stream():
        voiced = bytearray()
        speech_active = False
        t_start = 0.0
        t_last_voice = 0.0
        t_begin = 0.0
        last_level_log = time.time()

        try:
            while True:
                # Level meter (once per second)
                if time.time() - last_level_log >= 1.0:
                    # Expect 0.00..0.30 (speech typically > 0.02)
                    print(f"[Audio] level ~ {listener.last_rms:.3f}")
                    last_level_log = time.time()

                try:
                    chunk = listener.q.get(timeout=0.2)
                except queue.Empty:
                    chunk = None
                now = time.time()

                if chunk is not None:
                    # VAD decision on this 30ms frame
                    speech = listener.vad.is_speech(chunk, SAMPLE_RATE)
                    if speech:
                        if not speech_active:
                            speech_active = True
                            t_start = now
                            t_begin = now
                            voiced.clear()
                            print("[ASR] ▶ start")
                        voiced.extend(chunk)
                        t_last_voice = now

                if speech_active:
                    silence_ms = (now - t_last_voice) * 1000
                    dur_ms     = (t_last_voice - t_start) * 1000
                    hard_ms    = (now - t_begin) * 1000
                    if hard_ms > MAX_UTTER_SEC*1000 or (silence_ms >= SILENCE_END_MS and dur_ms >= MIN_UTTER_MS):
                        print("[ASR] ■ end → transcribe")
                        a = np.frombuffer(bytes(voiced), dtype=np.int16).astype(np.float32) / 32767.0
                        tmp = "/tmp/asr_stream.wav"
                        write_wav_pcm16(tmp, a, SAMPLE_RATE)

                        segs, info = model.transcribe(tmp, beam_size=1)
                        text = "".join(s.text for s in segs).strip()
                        lang = info.language or last_lang
                        if not text:
                            print("[ASR] (empty)")
                            speech_active = False
                            continue
                        print(f"[ASR] lang={lang} text={text}")

                        # manual language switch by content
                        if "中文" in text or "說中文" in text or "讲中文" in text:
                            lang = "zh"
                        elif re.search(r"\b(english|speak english)\b", text, re.I):
                            lang = "en"

                        system, prompt = build_prompt(text, lang)
                        reply = call_ollama(system, prompt) or (
                            "攝像頭目前沒有明顯物體。" if (is_vision_intent(text, lang) and str(lang).startswith("zh"))
                            else ("I don't see anything notable." if is_vision_intent(text, lang)
                                  else ("抱歉，我暫時無法回答。請再試一次。" if str(lang).startswith("zh")
                                        else "Sorry, I couldn't answer. Please try again."))
                        )
                        print(f"[BOT] ({lang}): {reply}")
                        tts_say(reply, lang)
                        last_lang = lang
                        speech_active = False
        except KeyboardInterrupt:
            print("\n[Assistant] Bye.")

if __name__ == "__main__":
    main()
