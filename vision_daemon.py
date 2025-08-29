#!/usr/bin/env python3
import time, json, cv2, os, sys
from ultralytics import YOLO

CAM_INDEX = 0
CONF = 0.35
OUT_PATH = "/tmp/detections.json"

def safe_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

def main():
    print("[Vision] Loading YOLOv8n …", file=sys.stderr)
    yolo = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[Vision] Camera open failed", file=sys.stderr)
        sys.exit(1)

    last_pub = 0.0
    print("[Vision] Running headless (no GUI). Press Ctrl+C to stop.", file=sys.stderr)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Vision] Frame grab failed", file=sys.stderr)
                break

            res = yolo.predict(frame, conf=CONF, verbose=False)
            det = []
            for r in res:
                for b in r.boxes:
                    c = int(b.cls[0].item()); label = r.names[c]
                    p = float(b.conf[0].item())
                    x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy().tolist())
                    det.append({"label":label,"conf":round(p,3),"box":[x1,y1,x2,y2]})

            now = time.time()
            if now - last_pub > 0.5:  # ~2 Hz
                safe_write_json(OUT_PATH, {"ts": now, "items": det})
                print(f"[Vision] {len(det)} objects -> {OUT_PATH}", file=sys.stderr)
                last_pub = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

if __name__ == "__main__":
    main()
