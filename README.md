# Jetson Voice Vision Assistant
Jetson Assistant, this will turn the Jetson Orin NX as voice assistant. It listens with Whisper, speaks with Pipere, detect objects with YOLOv8, and can chat with a local LLM (Ollama)

## Features 
<ol>
<li>🎤 ASR: Faster-Whisper (CPU, int8) with WebRTC VAD (always listening)</li>
<li>🔊 TTS: Piper voices (EN + ZH)</li>
<li>👀 Vision: YOLOv8n object detection; writes /tmp/detections.json</li>
<li>🤖 Chat: optional local LLM via Ollama (e.g., qwen2.5:1.5b-instruct)</li>
<li>🧠 Bilingual: auto language detect; replies in EN/中文</li>
<li>🧰 Headless: works over SSH; no X/Qt windows needed</li>
</ol>

## Hardware Requirement
<ol>
  <li>NVIDIA Jetson Orin Nano NX</li>
  <li>USB Mic Array v2 (or any USB Mic)</li>
  <li>Logitech Camera C270 (any UVC Webcam, high spec is better)</li>
  <li>Stable Power</li>
</ol>

## Quick Start

Clone this project
```bash
cd multilang_voice_vision
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Python deps
```bash
pip install ultralytics opencv-python \
            faster-whisper webrtcvad sounddevice numpy requests \
            piper-tts onnxruntime
sudo apt update && sudo apt install -y alsa-utils  # for aplay/arecord
```

Piper voices (EN + ZH)

1 Create the folder
```bash
mkdir -p voices
```
2 Download both files for each voice (model `.onnx` and config `.json` from the Piper models page
  -> `rhasspy/piper-voices` on Hugging Face. Example voices:
  * English (US) Ryan : `en_US-ryan.onnx` + `zh_CN-huayan.json`
    English
```bash
wget -O en_US-ryan.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
wget -O en_US-ryan.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
cp -n en_US-ryan.json en_US-ryan.onnx.json
```
 Chinese
```bash
wget -O zh_CN-huayan.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
wget -O zh_CN-huayan.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
cp -n zh_CN-huayan.json zh_CN-huayan.onnx.json
```
    
    
3. Add the `.onnx.json` filename Piper expects
```bash
cd voices
cp -n en_US-ryan.json      en_US-ryan.onnx.json
cp -n zh_CN-huayan.json    zh_CN-huayan.onnx.json
ls -1
# en_US-ryan.onnx  en_US-ryan.json  en_US-ryan.onnx.json
# zh_CN-huayan.onnx zh_CN-huayan.json zh_CN-huayan.onnx.json
```
Test the voice
```bash
echo '{"text":"Hello from Ryan","lang":"en"}' | python3 tts_cli.py
echo '{"text":"你好，我是华言","lang":"zh"}' | python3 tts_cli.py
```

4. Local LLM with Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
# Good small bilingual model:
ollama pull qwen2.5:1.5b-instruct
```

## Run 
Terminal A - Vision Object Detection
```bash
cd multilang_voice_vision
source .venv/bin/activate
python3 vision_daemon.py
# prints: [Vision] N objects -> /tmp/detections.json
```

Terminal B - Voice Input and Assistant
```bash
cd multilang_voice_vision
source .venv/bin/activate
# export MIC_INDEX=2
python3 assistant_vad_llm.py
```
Optional: select your mic by index once you know it (see Troubleshooting)

Speak normally
* General questions - talk with device
* "說中文/speak English" switch language
* 'What do you see? / 你看到什麼？it will describe objects from camera

##Configuration
Environment variables you can set before running
* `MIC_INDEX` force input device index (e.g., `export MIC_INDEX=2`)
* `WHISPER_SIZE` the default is `tiny` or `base` (more accurate, slower)
* `LLM_MODEL` we can use `qwen2.5:1.5b-instruct`, `phi3:mini`, `llama3.2:1b-instruct`

Tweakables inside `assistant_vad_llm.py`:
* `VAD_AGGR` (0-3), `SILENCE_END_MS`, `MAX_UTTER_SEC` - it responsiveness
* `num_predict` in `call_ollama_stream` - it will be limit reply length


## Troubleshooting
No audio /level stays `0.000`
List devices and pick your mic:
```bash
python3 - <<'PY'
import sounddevice as sd
for i,d in enumerate(sd.query_devices()):
    if d['max_input_channels']>0:
        print(f"{i:2d} | in={d['max_input_channels']:>2} | {d['name']}")
PY
```
After that choice the active mic (e.g., Mic Array)
```bash
export MIC_INDEX=<index>
python3 assistant_vad_llm.py
```
If still silent, run `alsamixer` -> F6 -> select your mic -> unmute Capture and raise volume.

### Piper/TTS runs but silent
Ensure the four voices files exist and names include `.onnx.json` (See step 3).
Test TTS:
```bash
echo '{"text":"Hello","lang":"en"}' | python3 tts_cli.py
```
File model.bin is incomplete
Whisper cache got corrupted; clear and retry
```bash
rm -rf ~/.cache/faster_whisper ~/.cache/ctranslate2
```
Our code already downloads into models/whisper/, so just run again.
### Camera: QT/XCB pluggin error
This is headless, use `vision_daemon.py` no GUI, it logs to console and writes `/tmp/detections.json`.
Ollama errors / slow
Check server
```bash
curl -s http://127.0.0.1:11434/api/tags
```

## Acknowledgements
* Whisper (faster-whisper/CTrasnlate2)
* Piper TTS and rhasspy/piper-voices
* Ultralytics YOLOv8
* Ollama for local LLM serving
* WebRTC VAD



## License

This repository’s code is released under the MIT License (see `LICENSE`).

**Third-party components and downloaded models/voices are licensed separately.**
Users must review and comply with the licenses of Ultralytics YOLO, Piper and its
voices, Whisper/faster-whisper, any Ollama models, and other dependencies.
No model/voice/weight files are included in this repo.




