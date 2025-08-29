# ROS2-Jetson-Assistant
ROS2 Jetson Assistant, this will turn the Jetson Orin NX as voice assistant. It listens with Whisper, speaks with Pipere, detect objects with YOLOv8, and can chat with a local LLM (Ollama)

Features
🎤 ASR: Faster-Whisper (CPU, int8) with WebRTC VAD (always listening)
🔊 TTS: Piper voices (EN + ZH)
👀 Vision: YOLOv8n object detection; writes /tmp/detections.json
🤖 Chat: optional local LLM via Ollama (e.g., qwen2.5:1.5b-instruct)
🧠 Bilingual: auto language detect; replies in EN/中文
🧰 Headless: works over SSH; no X/Qt windows needed

Hardware Requirement
1. NVIDIA Jetson Orin Nano NX
2. USB Mic Array v2 (or any USB Mic)
3. Logitech Camera C270 (any UVC Webcam, high spec is better)
4. Stable Power
