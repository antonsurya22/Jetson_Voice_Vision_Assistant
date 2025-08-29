#!/usr/bin/env python3
import sys, json, time, os, wave
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
SECONDS = float(os.getenv("ASR_SECONDS", "4"))
MODEL_SIZE = os.getenv("WHISPER_SIZE", "small")
DEVICE = "cpu"
COMPUTE = "int8"

# If your Mic Array isn't the default input, set its index here (check with sd.query_devices()).
# sd.default.device = (None, 2)   # (output_device, input_device_index)

def write_wav_pcm16(path, audio_f32, sr=SAMPLE_RATE):
    """Write mono float32 [-1,1] to 16-bit PCM WAV."""
    # ensure mono
    if audio_f32.ndim > 1:
        audio_f32 = audio_f32[:, 0]
    # clip & convert to int16
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())

def main():
    print(f"[ASR] Recording {SECONDS:.1f}s…", file=sys.stderr)
    audio = sd.rec(int(SECONDS*SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="float32")
    sd.wait()
    tmp = "/tmp/asr.wav"
    write_wav_pcm16(tmp, audio)

    print(f"[ASR] Loading Whisper: {MODEL_SIZE}", file=sys.stderr)
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
    segments, info = model.transcribe(tmp, beam_size=1)
    text = "".join(s.text for s in segments).strip()
    lang = info.language or "en"
    out = {"text": text, "lang": lang, "ts": time.time()}
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
