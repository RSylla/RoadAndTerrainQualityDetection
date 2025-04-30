import os
import time
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import nms
import matplotlib.pyplot as plt
# Pretrained weights for backbone
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
import pandas as pd

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE       = 320
BATCH_SIZE     = 32
EPOCHS         = 150
LR             = 1e-3
WEIGHT_DECAY   = 5e-4
NUM_WORKERS    = 4
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH    = 0.25
IOU_THRESH     = 0.5
MODEL_NAME     = 'mobilenet_v3_large'

CLASS_NAMES    = [
    "background", "D00", "D01", "D10", "D11", "D20", "D40",
    "D43", "D44", "Repair", "Block crack", "D50", "D0w0"
]

NUM_CLASSES    =len(CLASS_NAMES)  # RDD2022 damage classes
# Class names including background at index 0
# YOLOv10 style augmentation configurations
MOSAIC_PROB    = 0.5
MIXUP_PROB     = 0.2
HSV_HUE        = 0.015
HSV_SAT        = 0.7
HSV_VAL        = 0.4
FLIP_PROB      = 0.5

# Directories
DATA_ROOT      = 'dataset'
TRAIN_IMG_DIR  = os.path.join(DATA_ROOT, 'images', 'train')
TRAIN_LBL_DIR  = os.path.join(DATA_ROOT, 'xml',   'train')  # PascalVOC XML labels
VAL_IMG_DIR    = os.path.join(DATA_ROOT, 'images', 'val')
VAL_LBL_DIR    = os.path.join(DATA_ROOT, 'xml',   'val')    # PascalVOC XML labels
TEST_IMG_DIR   = os.path.join(DATA_ROOT, 'images', 'test')
TEST_LBL_DIR   = os.path.join(DATA_ROOT, 'xml',   'test')   # PascalVOC XML labels

RESULTS_DIR    = f'SSD_{MODEL_NAME}/results'
MODEL_DIR      = f'SSD_{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------
# Utility Function: load image and labels
# ---------------------------
def load_image_and_labels(img_path, lbl_path, img_size):
    import xml.etree.ElementTree as ET
    img = cv2.imread(img_path)
    assert img is not None, f"Missing image {img_path}"
    h0, w0 = img.shape[:2]
    boxes = []
    if os.path.isfile(lbl_path):
        tree = ET.parse(lbl_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name in CLASS_NAMES:
                cls = CLASS_NAMES.index(cls_name)
            else:
                continue
            bnd = obj.find('bndbox')
            x1 = float(bnd.find('xmin').text)
            y1 = float(bnd.find('ymin').text)
            x2 = float(bnd.find('xmax').text)
            y2 = float(bnd.find('ymax').text)
            if not x2 - x1 > 0:
                x1 -= 3
                x2 += 3
            if not y2 - y1 > 0:
                y1 -= 3
                y2 += 3
            boxes.append([x1, y1, x2, y2, cls])
    boxes = np.array(boxes, dtype=np.float32)
    # Resize image and scale boxes
    img = cv2.resize(img, (img_size, img_size))
    if boxes.size:
        scale_x = img_size / w0
        scale_y = img_size / h0
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y
    return img, boxes


# ---------------------------
# Dataset with YOLOv10-style Augmentations
# ---------------------------
class RDDDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=IMG_SIZE, augment=False):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.augment = augment
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        def load(idx):
            img_path = os.path.join(self.img_dir, self.files[idx])
            lbl_path = os.path.join(self.lbl_dir, os.path.splitext(self.files[idx])[0] + '.xml')  # use PascalVOC XML
            return load_image_and_labels(img_path, lbl_path, self.img_size)

        # Mosaic augmentation
        if self.augment and random.random() < MOSAIC_PROB:
            mosaic_img = np.full((self.img_size*2, self.img_size*2, 3), 114, dtype=np.uint8)
            xc = random.randint(self.img_size//2, int(self.img_size*1.5))
            yc = random.randint(self.img_size//2, int(self.img_size*1.5))
            all_boxes = []
            ids = [idx] + random.sample(range(len(self)), 3)
            for i, idm in enumerate(ids):
                img, boxes = load(idm)
                h, w = img.shape[:2]
                if i == 0:
                    x1a, y1a, x2a, y2a = max(xc-w, 0), max(yc-h, 0), xc, yc
                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, self.img_size*2), yc
                elif i == 2:
                    x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(yc+h, self.img_size*2)
                else:
                    x1a, y1a, x2a, y2a = xc, yc, min(xc+w, self.img_size*2), min(yc+h, self.img_size*2)
                dx, dy = x2a - x1a, y2a - y1a
                x1b, y1b = max(0, -x1a), max(0, -y1a)
                x2b, y2b = x1b + dx, y1b + dy
                sub = img[y1b:y2b, x1b:x2b]
                if sub.size:
                    mosaic_img[y1a:y2a, x1a:x2a] = cv2.resize(sub, (dx, dy))
                    if boxes.size:
                        b = boxes.copy()
                        b[:, [0,2]] = b[:, [0,2]] - x1b + x1a
                        b[:, [1,3]] = b[:, [1,3]] - y1b + y1a
                        all_boxes.append(b)
            x_start = random.randint(0, self.img_size)
            y_start = random.randint(0, self.img_size)
            img = mosaic_img[y_start:y_start+self.img_size, x_start:x_start+self.img_size]
            if all_boxes:
                boxes = np.concatenate(all_boxes, axis=0)
                boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, self.img_size)
                boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, self.img_size)
                boxes = boxes[(boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])]
            else:
                boxes = np.zeros((0,5), dtype=np.float32)
        else:
            img, boxes = load(idx)

        # MixUp augmentation
        if self.augment and random.random() < MIXUP_PROB:
            img2, boxes2 = load(random.randrange(len(self)))
            img = ((img.astype(np.float32) + img2.astype(np.float32)) / 2).astype(np.uint8)
            if boxes2.size:
                boxes = np.vstack((boxes, boxes2)) if boxes.size else boxes2.copy()

        # HSV color-space augmentation
        if self.augment:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-HSV_HUE*180, HSV_HUE*180)) % 180
            hsv[..., 1] *= random.uniform(1-HSV_SAT, 1+HSV_SAT)
            hsv[..., 2] *= random.uniform(1-HSV_VAL, 1+HSV_VAL)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Horizontal flip
        if self.augment and random.random() < FLIP_PROB:
            img = cv2.flip(img, 1)
            if boxes.size:
                boxes[:, 0], boxes[:, 2] = self.img_size - boxes[:, 2], self.img_size - boxes[:, 0]

        # To tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.copy().transpose(2,0,1)).float() / 255.0

        # Target
        if boxes.size:
            coords = torch.from_numpy(boxes[:, :4]).float()  # [xmin, ymin, xmax, ymax]
            labels = torch.from_numpy(boxes[:, 4]).long()  # use label indices matching CLASS_NAMES  # shift labels to 1..NUM_CLASSES for background=0
        else:
            coords = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return img, {'boxes': coords, 'labels': labels}


# Collate function for DataLoader
def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

# Create datasets and loaders
train_ds = RDDDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, augment=True)
val_ds   = RDDDataset(VAL_IMG_DIR, VAL_LBL_DIR, IMG_SIZE, augment=False)
test_ds  = RDDDataset(TEST_IMG_DIR, TEST_LBL_DIR, IMG_SIZE, augment=False)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
#test_loader   = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# Initialize model, optimizer, scheduler
model = ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    num_classes=NUM_CLASSES
).to(DEVICE)  # use pre-trained backbone to enable meaningful losses  # num_classes includes background  # include background class.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2)

# Training and validation
train_losses, val_losses = [], []
best_val = float('inf')
for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{'boxes': t['boxes'].to(DEVICE), 'labels': t['labels'].to(DEVICE)} for t in tgts]
        optimizer.zero_grad()
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item() * len(imgs)
    train_loss = running / len(train_ds)
    train_losses.append(train_loss)

    # Validation (use train mode to get losses)
    model.train()
    val_running = 0.0
    with torch.no_grad():
        for imgs, tgts in tqdm(val_loader, desc='Val'):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{'boxes': t['boxes'].to(DEVICE), 'labels': t['labels'].to(DEVICE)} for t in tgts]
            loss_dict = model(imgs, targets)
            val_running += sum(loss_dict.values()).item() * len(imgs)
    val_loss = val_running / len(val_ds)
    val_losses.append(val_loss)
    model.eval()
    print(f"Epoch {epoch}/{EPOCHS} – Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best.pth'))
# Save final model
torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'final.pth'))

# Plot and save loss curve
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.jpg'), dpi=300)


# Evaluation function
from collections import defaultdict

def compute_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.0

def evaluate(dataset):
    model.eval()
    tp = fp = total_gt = 0
    with torch.no_grad():
        for img, tgt in tqdm(dataset, desc='Eval'):
            out = model([img.to(DEVICE)])[0]
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            keep = scores > CONF_THRESH
            boxes = boxes[keep]
            scores = scores[keep]
            if len(boxes):
                idxs = nms(torch.tensor(boxes), torch.tensor(scores), IOU_THRESH)
                boxes = boxes[idxs]
            # Ground truths in [xmin, ymin, xmax, ymax]
            gts = tgt['boxes'].cpu().numpy()  # retrieved in same format
            total_gt += len(gts)
            matched = set()
            for b in boxes:
                ious = [compute_iou(b, gt) for gt in gts]
                if ious and max(ious) >= IOU_THRESH:
                    j = ious.index(max(ious))
                    if j not in matched:
                        tp +=1
                        matched.add(j)
                    else:
                        fp +=1
                else:
                    fp +=1
    precision = tp/(tp+fp+1e-6)
    recall = tp/(total_gt+1e-6)
    mAP50 = precision
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return {'mAP50': mAP50, 'precision': precision, 'recall': recall, 'f1': f1}


# Test evaluation and save
metrics = evaluate(test_ds)
plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values())
plt.xticks(rotation=45)
plt.title('Test Metrics')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'metrics.jpg'), dpi=300)
pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, 'metrics.csv'), index=False)


# Inference on 4 random samples
indices = list(range(len(test_ds)))
samples = random.sample(indices, 4)
fig, axs = plt.subplots(2,2, figsize=(6,6))
for ax, idx in zip(axs.flatten(), samples):
    img, _ = test_ds[idx]
    out = model([img.to(DEVICE)])[0]
    boxes = out['boxes'].cpu().numpy()
    scores = out['scores'].cpu().numpy()
    labels = out['labels'].cpu().numpy()
    keep = scores > CONF_THRESH
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if len(boxes):
        idxs = nms(torch.tensor(boxes), torch.tensor(scores), IOU_THRESH)
        boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]
    im = (img.numpy().transpose(1,2,0)*255).astype(np.uint8)[..., ::-1]
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, b)
        cls_name = CLASS_NAMES[int(l)] if int(l) < len(CLASS_NAMES) else str(int(l))
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(im, f"{cls_name} {s:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    ax.imshow(im)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'inference.jpg'), dpi=300)