import os
import random
import cv2
import time
import torch
import numpy as np

from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from ultralytics import YOLO

# ─────────────────────────── Settings ───────────────────────────
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

IMG_SIZE         = 512
TEST_IMG_DIR     = 'dataset/images/test'
NUM_CLASSES      = 13

CKPT_FILE            = 'effdet_best.ckpt'
EFF_WEIGHTS_PATH     = os.path.join('EfficientDet_D0', CKPT_FILE)
YOLO_WEIGHTS         = 'EfficientDet_D0/yolov12s.pt'
SSDLITE_WEIGHTS_PATH = os.path.join('EfficientDet_D0', 'ssdlite_mobilenetv3large.pth')

WARMUP_RUNS      = 10
MEASURE_RUNS     = 3
NUM_SAMPLES      = 20

# ───────────────────────── Utilities ──────────────────────────
def load_image(path, img_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    t = torch.from_numpy(img).permute(2,0,1).float().div(255.0)
    return t.unsqueeze(0)

def measure_latency(model, img_tensor):
    model.eval()
    img = img_tensor.to(DEVICE).half()
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(img)
        for _ in range(MEASURE_RUNS):
            starter.record()
            _ = model(img)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    arr = np.array(times, dtype=np.float32)
    return arr.mean()

# ───────────────────────── Load Models ─────────────────────────
# EfficientDet-D0
conf = get_efficientdet_config('tf_efficientdet_d0')
conf.image_size  = (IMG_SIZE, IMG_SIZE)
conf.num_classes = NUM_CLASSES
eff = EfficientDet(conf, pretrained_backbone=True)
ckpt = torch.load(EFF_WEIGHTS_PATH, map_location=DEVICE)
sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
eff.load_state_dict(sd, strict=False)
eff = DetBenchPredict(eff).half().to(DEVICE)

# YOLO v12s
yolo = YOLO(YOLO_WEIGHTS)
yolo_model = yolo.model.half().to(DEVICE)

# SSDlite
ssd = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=NUM_CLASSES)
ssd.load_state_dict(torch.load(SSDLITE_WEIGHTS_PATH, map_location=DEVICE))
ssd = ssd.eval().half().to(DEVICE)

# ──────────────────── Sample Test Images ───────────────────────
all_files = sorted(f for f in os.listdir(TEST_IMG_DIR)
                   if f.lower().endswith(('.jpg','.jpeg','.png')))
random.seed(42)
files = random.sample(all_files, min(NUM_SAMPLES, len(all_files)))
img_tensors = [load_image(os.path.join(TEST_IMG_DIR, f), IMG_SIZE) for f in files]

# ────────────────────────── Warm-up ─────────────────────────────
print("Warming up models...")
with torch.no_grad():
    dummy = img_tensors[0].to(DEVICE).half()
    for _ in range(WARMUP_RUNS):
        _ = eff(dummy)
        _ = yolo_model(dummy)
        _ = ssd(dummy)

# ─────────────────────── Measure Latencies ──────────────────────
eff_times, yolo_times, ssd_times = [], [], []

print("Measuring on random subset:")
for fname, img in zip(files, img_tensors):
    te = measure_latency(eff, img)
    ty = measure_latency(yolo_model, img)
    ts = measure_latency(ssd, img)
    eff_times.append(te)
    yolo_times.append(ty)
    ssd_times.append(ts)
    print(f"{fname:30} → EfficientDet: {te:6.2f} ms | YOLO: {ty:6.2f} ms | SSD: {ts:6.2f} ms")

# ────────────────────────── Summary ────────────────────────────
print("\n=== Overall Avg (± std) ===")
print(f"EfficientDet-D0 : {np.mean(eff_times):6.2f} ms ± {np.std(eff_times):.2f}")
print(f"YOLO v12s       : {np.mean(yolo_times):6.2f} ms ± {np.std(yolo_times):.2f}")
print(f"SSDLite         : {np.mean(ssd_times):6.2f} ms ± {np.std(ssd_times):.2f}")
