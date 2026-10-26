import glob
import statistics
from ultralytics import YOLO

W = r"runs\detect\runs\visdrone_s\weights"
IMAGES = sorted(glob.glob(r"datasets\VisDrone\images\val\*.jpg"))[:100]
CONFIGS = [
    ("PyTorch FP32", f"{W}\\best.pt", "cpu"),
    ("PyTorch FP32", f"{W}\\best.pt", 0),
    ("ONNX FP32",    f"{W}\\best.onnx", "cpu"),
]

print("| Model | Device | Inference ms | Total ms (pre+inf+post) | FPS |")
print("|---|---|---|---|---|")
for name, path, dev in CONFIGS:
    m = YOLO(path)
    inf, tot = [], []
    for i, img in enumerate(IMAGES):
        r = m.predict(img, imgsz=640, conf=0.22, device=dev, verbose=False)[0]
        if i < 10:   # warmup
            continue
        s = r.speed
        inf.append(s["inference"])
        tot.append(s["preprocess"] + s["inference"] + s["postprocess"])
    mi, mt = statistics.mean(inf), statistics.mean(tot)
    label = "GPU (RTX 3050)" if dev == 0 else "CPU"
    print(f"| {name} | {label} | {mi:.1f} | {mt:.1f} | {1000/mt:.1f} |")