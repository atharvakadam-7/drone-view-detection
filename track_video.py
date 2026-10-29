import os
import sys
import time
import cv2
import torch
from ultralytics import YOLO

MODEL = os.environ.get("MODEL_PATH", "weights/best.pt")
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SRC = sys.argv[1] if len(sys.argv) > 1 else "drone.mp4"
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 640
OUT = f"{SRC.rsplit('.', 1)[0]}_tracked_{IMGSZ}.mp4"
CONF = 0.22
W, H = 1280, 720  # every frame is resized to this before inference

model = YOLO(MODEL)
cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    raise SystemExit(f"Could not open {SRC}. Check the filename and that it's in this folder.")

fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))

ids, times = set(), []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (W, H))
    t0 = time.perf_counter()
    r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                    conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    times.append(time.perf_counter() - t0)
    if r.boxes.id is not None:
        ids.update(r.boxes.id.int().tolist())
    out_frame = r.plot(labels=False, conf=False)
    cv2.putText(out_frame, f"{1.0 / times[-1]:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    writer.write(out_frame)

cap.release()
writer.release()

if not times:
    raise SystemExit("Read 0 frames. The video may be corrupt or use an unsupported codec.")

steady = times[10:] or times  # skip warmup frames
avg = sum(steady) / len(steady)
print(f"video: {SRC} | imgsz: {IMGSZ} | input resized to {W}x{H}")
print(f"frames: {len(times)}")
print(f"avg latency: {avg*1000:.1f} ms/frame -> {1/avg:.1f} FPS (excl. first 10 frames)")
print(f"unique track IDs: {len(ids)}  (overcounts: ID switches create new IDs)")