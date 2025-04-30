import os
import xml.etree.ElementTree as ET
from glob import glob

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import json, random
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -- User configuration --
# Set these to your dataset root and class list
DATA_ROOT = 'dataset'
CLASS_NAMES = [
    'background','D00','D01','D10','D11','D20','D40',
    'D43','D44','Repair','Block crack','D50','D0w0'
]
NUM_CLASSES = len(CLASS_NAMES)  # replace with your actual class names


def create_model(num_classes, img_size, architecture='efficientdet_d0'):
    config = get_efficientdet_config(architecture)
    config.num_classes = num_classes
    config.image_size = (img_size, img_size)  # tuple for FPN compatibility
    model = EfficientDet(config, pretrained_backbone=True)
    model = DetBenchTrain(model, config)
    return model


def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


class EfficientDetDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transforms = transforms
        self.files = sorted(glob(os.path.join(img_dir, '*.*')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_id = idx
        img_path = self.files[idx]
        xml_path = os.path.join(
            self.lbl_dir,
            os.path.splitext(os.path.basename(img_path))[0] + '.xml'
        )
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load raw annotations
        raw_bboxes, raw_labels = [], []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            for obj in tree.findall('object'):
                cls = obj.find('name').text
                if cls not in CLASS_NAMES:
                    continue
                raw_labels.append(CLASS_NAMES.index(cls))
                bnd = obj.find('bndbox')
                x1 = float(bnd.find('xmin').text)
                y1 = float(bnd.find('ymin').text)
                x2 = float(bnd.find('xmax').text)
                y2 = float(bnd.find('ymax').text)
                if x2 <= x1 or y2 <= y1:
                    x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
                    x2, y2 = x2 + 1, y2 + 1
                raw_bboxes.append([x1, y1, x2, y2])

        # Apply transforms or use raw data
        if self.transforms:
            transformed = self.transforms(
                image=img, bboxes=raw_bboxes, labels=raw_labels
            )
            image = transformed['image']
            tb, tl = transformed['bboxes'], transformed['labels']
            if len(tb) > 0:
                bboxes = np.array(tb, dtype=np.float32)
                labels = np.array(tl, dtype=np.int64)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
        else:
            image = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            if len(raw_bboxes) > 0:
                bboxes = np.array(raw_bboxes, dtype=np.float32)
                labels = np.array(raw_labels, dtype=np.int64)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

        # Convert bboxes from [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
        if bboxes.size:
            bboxes = bboxes[:, [1, 0, 3, 2]]

        target = {
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'img_size': torch.tensor([image.shape[1], image.shape[2]]),
            'img_scale': torch.tensor([1.0]),
            'image_id': torch.tensor([image_id])
        }
        return image, target, image_id


class EfficientDetDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root=DATA_ROOT,
                 train_transforms=None,
                 valid_transforms=None,
                 batch_size=8,
                 num_workers=4):
        super().__init__()
        self.data_root = data_root
        self.train_transforms = train_transforms or get_train_transforms()
        self.valid_transforms = valid_transforms or get_valid_transforms()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        train_img = os.path.join(self.data_root, 'images', 'train')
        train_lbl = os.path.join(self.data_root, 'xml', 'train')
        val_img = os.path.join(self.data_root, 'images', 'val')
        val_lbl = os.path.join(self.data_root, 'xml', 'val')

        if stage in (None, 'fit'):
            self.train_dataset = EfficientDetDataset(
                img_dir=train_img,
                lbl_dir=train_lbl,
                transforms=self.train_transforms
            )
        if stage in (None, 'fit', 'validate'):
            self.val_dataset = EfficientDetDataset(
                img_dir=val_img,
                lbl_dir=val_lbl,
                transforms=self.valid_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = zip(*batch)
        images = torch.stack(images).float()

        boxes = [t['bboxes'] for t in targets]
        labels = [t['labels'] for t in targets]
        img_size = torch.stack([t['img_size'] for t in targets]).float()
        img_scale = torch.stack([t['img_scale'] for t in targets]).float()

        annotations = {
            'bbox': boxes,
            'cls': labels,
            'img_size': img_size,
            'img_scale': img_scale
        }
        return images, annotations, targets, image_ids


class EfficientDetModel(pl.LightningModule):
    def __init__(self,
                 num_classes=len(CLASS_NAMES),
                 img_size=512,
                 lr=2e-4,
                 architecture='efficientdet_d0'):
        super().__init__()
        self.training_step_outputs = []
        self.save_hyperparameters()
        self.model = create_model(num_classes, img_size, architecture)

    def forward(self, images, annotations=None):
        return self.model(images, annotations)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, _ = batch
        losses = self.model(images, annotations)
        loss = losses['loss']
        self.log_dict({
            'train_loss': losses['loss'],
            'train_box_loss': losses['box_loss'],
            'train_class_loss': losses['class_loss'],
        }, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        self.log_dict({
            'val_loss': outputs['loss'],
            'val_box_loss': outputs['box_loss'],
            'val_class_loss': outputs['class_loss'],
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {'outputs': outputs, 'targets': targets, 'image_ids': image_ids}
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory


# 1. ——— SETUP DATA & MODEL —————
data_module = EfficientDetDataModule(
    data_root=DATA_ROOT,
    batch_size=32,
    num_workers=0
)

model = EfficientDetModel(
    num_classes=len(CLASS_NAMES),
    img_size=512,
    lr=2e-4,
    architecture='efficientdet_d0'
)

# 2. ——— TRAIN + VALIDATE —————
ckpt_callback = ModelCheckpoint(
    dirpath='efficientdet_d0_best/checkpoints/',
    filename= 'effdet-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss',
    mode='min'
)

trainer = Trainer(
    accelerator='cuda',    # or replace with accelerator='auto' for TPU/CPU
    max_epochs=150,
    precision=16,          # mixed-precision training
    callbacks=[ckpt_callback],
)

# run!
trainer.fit(model, datamodule=data_module)
trainer.validate(model, datamodule=data_module)


# 3. ——— INFERENCE —————
# Wrap the backbone for fast inference
#load the best checkpoint
# 0) Configuration
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf_thresh = 0.2
best_model_path = ckpt_callback.best_model_path # path to the best checkpoint in format 'efficientdet_d0_best/checkpoints/effdet-{epoch:02d}-{val_loss:.2f}.ckpt'
ckpt_path   =  os.path(best_model_path)
    
os.makedirs('efficientdet_d0_best/coco_eval', exist_ok=True)

# 1) Load Lightning checkpoint
model = EfficientDetModel.load_from_checkpoint(ckpt_path)
model = model.to(device).eval()

# 2) Wrap for inference
#    our LightningModule.model is a DetBenchTrain(config), so unwrap to get the raw EfficientDet
base_effdet = model.model.model
predictor  = DetBenchPredict(base_effdet).to(device).eval()

def infer_image(img_path, conf_thresh=0.2):
    # load + preprocess
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    tfm = get_valid_transforms(target_img_size=512)
    inp = tfm(image=arr, bboxes=[[0,0,1,1]], labels=[0])['image'].unsqueeze(0)
    
    # forward
    with torch.no_grad():
        det = predictor(inp.to(model.device))[0]  # (N, 6) tensor: [x1,y1,x2,y2,score,class]
    # filter by confidence
    det = det[det[:,4] > conf_thresh].cpu().numpy()
    boxes = det[:, :4]
    scores = det[:, 4]
    classes = det[:, 5].astype(int)
    return boxes, scores, classes

# Plot sample images with predictions and confidence in a 1x5 figure and save as a JPG.
# fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('EfficientDet Predictions', fontsize=16)
for idx, img_f in enumerate(os.listdir(os.path.join(DATA_ROOT, 'images', 'val'))[:5]):
    b, s, c = infer_image(os.path.join(DATA_ROOT, 'images', 'val', img_f))
    img = Image.open(os.path.join(DATA_ROOT, 'images', 'val', img_f)).convert('RGB')
    
    # Plot the image
    axes[idx].imshow(img)
    axes[idx].axis('off')
    axes[idx].set_title(f"{img_f}")
    
    # Add predictions
    for bb, sc, cl in zip(b, s, c):
        x1, y1, x2, y2 = bb
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        axes[idx].add_patch(rect)
        axes[idx].text(x1, y1 - 5, f"{CLASS_NAMES[cl]}: {sc:.2f}", color='red', fontsize=8, backgroundcolor='white')

# Save the figure as a JPG
plt.tight_layout()
plt.savefig("predictions.jpg", format='jpg')
plt.show()


# --- 4. COCO-STYLE METRICS (fixed) ---
device = model.device  # e.g. 'cuda'

# 1) Build the inference bench once
# Wrap the underlying EfficientDet for inference
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf_thresh = 0.2
os.makedirs('efficientdet_d0_best/coco_eval', exist_ok=True)

# 1) Load Lightning checkpoint
model = EfficientDetModel.load_from_checkpoint(ckpt_path)
model = model.to(device).eval()

# 2) Wrap for inference
#    our LightningModule.model is a DetBenchTrain(config), so unwrap to get the raw EfficientDet
base_effdet = model.model.model
predictor  = DetBenchPredict(base_effdet).to(device).eval()
# 2) Prepare output containers
coco_dataset = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i+1, "name": name} for i, name in enumerate(CLASS_NAMES)],
}
coco_detections = []

ann_id = 1
img_id_set = set()

# 3) Loop over validation set
for batch in tqdm(data_module.val_dataloader(), desc='Batch'):
    images, annotations, targets, image_ids = batch
    images = images.to(device)

    with torch.no_grad():
        batch_dets = predictor(images)           # Tensor[N, K, 6]
    batch_dets = batch_dets.cpu().numpy()

    for i, img_id in enumerate(image_ids):
        # — image info —
        if img_id not in img_id_set:
            h, w = targets[i]['img_size'].tolist()
            coco_dataset["images"].append({
                "id": img_id,
                "width": int(w),
                "height": int(h),
                "file_name": f"{img_id}.jpg",
            })
            img_id_set.add(img_id)

        # — ground-truth annotations —
        gt_boxes = targets[i]["bboxes"][:, [1,0,3,2]].tolist()  # [xmin,ymin,xmax,ymax]
        gt_labels = targets[i]["labels"].tolist()
        for box, cls in zip(gt_boxes, gt_labels):
            x1,y1,x2,y2 = box
            coco_dataset["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls)+1,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "area": float((x2-x1)*(y2-y1)),
                "iscrowd": 0,
            })
            ann_id += 1

        # — predictions —
        dets = batch_dets[i]  # shape (K,6): [x1,y1,x2,y2,score,class]
        for x1,y1,x2,y2,score,cls in dets:
            if score <  conf_thresh:
                continue
            coco_detections.append({
                "image_id": img_id,
                "category_id": int(cls)+1,
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(score),
            })

# 4) Write JSONs
os.makedirs("coco_eval", exist_ok=True)
gt_json   = "coco_eval/instances_val_gt.json"
pred_json = "coco_eval/instances_val_preds.json"
with open(gt_json,   "w") as f: json.dump(coco_dataset,  f)
with open(pred_json, "w") as f: json.dump(coco_detections, f)
print("Wrote:", gt_json, pred_json)

# -- 1) Run COCOeval and pull out metrics --
coco_gt   = COCO(gt_json)
coco_dt   = coco_gt.loadRes(pred_json)
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()  # prints full table

stats = coco_eval.stats  # numpy array of length 12
metrics = {
    'mAP@[.50:.95]': float(stats[0]),
    'mAP@.50':       float(stats[1]),
    'mAP@.75':       float(stats[2]),
    'AR@100':        float(stats[8]),
}
# we approximate “precision” by AP@.50 and “recall” by AR@100
precision = metrics['mAP@.50']
recall    = metrics['AR@100']
f1        = 2 * precision * recall / (precision + recall + 1e-6)
metrics.update({'precision': precision, 'recall': recall, 'f1': f1})

os.makedirs('coco_eval', exist_ok=True)
with open('coco_eval/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print('→ Saved metrics:', metrics)

# -- 2) Draw 4 random validation samples with boxes+scores --
val_dir = os.path.join(DATA_ROOT, 'images', 'val')
all_imgs = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg','.png'))])
samples  = random.sample(all_imgs, min(4, len(all_imgs)))

fig, axs = plt.subplots(2, 2, figsize=(12,12))
axs = axs.flatten()
for ax, fname in zip(axs, samples):
    # load & original size
    img_path = os.path.join(val_dir, fname)
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h0, w0 = orig.shape[:2]

    # preprocess & predict
    arr = cv2.resize(orig, (512,512)).astype(np.float32)
    tfm = get_valid_transforms(512)
    inp = tfm(image=arr, bboxes=[[0,0,1,1]], labels=[0])['image'].unsqueeze(0)
    with torch.no_grad():
        dets = predictor(inp.to(device))[0].cpu().numpy()  # (K,6)

    # plot
    ax.imshow(orig)
    for x1,y1,x2,y2,score,cls in dets:
        if score < conf_thresh: continue
        # rescale back
        sx, sy = w0/512, h0/512
        x1n, y1n = int(x1*sx), int(y1*sy)
        w, h    = int((x2-x1)*sx), int((y2-y1)*sy)
        rect = plt.Rectangle((x1n,y1n), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1n, y1n, f"{CLASS_NAMES[int(cls)]}:{score:.2f}", 
                bbox=dict(facecolor='black', alpha=0.5), fontsize=10)
    ax.axis('off')
    ax.set_title(fname)

plt.tight_layout()
out_png = 'coco_eval/pred_samples.png'
plt.savefig(out_png)
print(f'→ Saved sample predictions to {out_png}')
plt.show()


metrics = {
    # 'mAP@[.50:.95]': float(stats[0]),
    'mAP@.50':       float(stats[1]),
    # 'mAP@.75':       float(stats[2]),
    # 'AR@100':        float(stats[8]),
    'precision':     precision,
    'recall':        recall,
    'f1':            f1,
    'infTime':      float(stats[11]),
}
plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values())
plt.xticks(rotation=45)
plt.title('EfficientDet D0 Metrics')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join('effdet_metrics.jpg'), dpi=300)