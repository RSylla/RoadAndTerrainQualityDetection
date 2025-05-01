# video_classifier.py
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from ultralytics import YOLO
import os

# 27 RSCD classes
RSCD_CLASSES = [
    "dry-asphalt-smooth","dry-asphalt-slight","dry-asphalt-severe",
    "dry-concrete-smooth","dry-concrete-slight","dry-concrete-severe",
    "dry-dirt/mud","dry-gravel",
    "wet-asphalt-smooth","wet-asphalt-slight","wet-asphalt-severe",
    "wet-concrete-smooth","wet-concrete-slight","wet-concrete-severe",
    "wet-dirt/mud","wet-gravel",
    "water-asphalt-smooth","water-asphalt-slight","water-asphalt-severe",
    "water-concrete-smooth","water-concrete-slight","water-concrete-severe",
    "water-dirt/mud","water-gravel",
    "fresh snow","melted snow","ice"
]
NUM_CLASSES = len(RSCD_CLASSES)

# Build your classifier head
def create_classifier(feat_dim, num_classes, dropout_rate=0.2, use_batchnorm=True):
    layers = []
    if dropout_rate > 0:
        layers.append(nn.Dropout(p=dropout_rate))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(feat_dim))
    layers.append(nn.Linear(feat_dim, num_classes))
    return nn.Sequential(*layers)

# # # Wrap MobileNetV3-Large backbone + your head
# def create_rscd_model(num_classes):
#     base = models.mobilenet_v3_large(pretrained=True)
#     backbone = base.features       # outputs 960 channels
#     classifier = create_classifier(960, num_classes)
#     class Net(nn.Module):
#         def __init__(self, backbone, classifier):
#             super().__init__()
#             self.backbone   = backbone
#             self.classifier = classifier
#         def forward(self, x):
#             feats = self.backbone(x)
#             feats = F.adaptive_avg_pool2d(feats, (1,1))
#             feats = torch.flatten(feats, 1)
#             logits = self.classifier(feats)
#             return logits
#     return Net(backbone, classifier)


# Wrap Efficientdet backbone + your head
def create_rscd_model(num_classes):
    base_model = models.efficientnet_b0(pretrained=True)
    in_features = base_model.classifier[1].in_features  # typically 1280
    base_model.classifier[1] = nn.Linear(in_features, num_classes)
    backbone = base_model.features
    classifier = create_classifier(in_features, num_classes)
    class Net(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone   = backbone
            self.classifier = classifier
        def forward(self, x):
            feats = self.backbone(x)
            if feats.dim() == 4:
                feats = F.adaptive_avg_pool2d(feats, (1,1))
            feats = torch.flatten(feats, start_dim=1)
            logits = self.classifier(feats)
            return logits
    return Net(backbone, classifier)

# Wrap Efficientdet backbone + your head
# def create_rscd_model(num_classes):
#     base_model = models.mnasnet1_3(pretrained=True)
#     in_features = base_model.classifier[1].in_features  # typically 1280
#     base_model.classifier[1] = nn.Linear(in_features, num_classes)
#     backbone = base_model.layers
#     classifier = create_classifier(in_features, num_classes)
#     class Net(nn.Module):
#         def __init__(self, backbone, classifier):
#             super().__init__()
#             self.backbone   = backbone
#             self.classifier = classifier
#         def forward(self, x):
#             feats = self.backbone(x)
#             if feats.dim() == 4:
#                 feats = F.adaptive_avg_pool2d(feats, (1,1))
#             feats = torch.flatten(feats, start_dim=1)
#             logits = self.classifier(feats)
#             return logits
#     return Net(backbone, classifier)

# Load fine-tuned weights
def load_rscd_model(weights_path, device):
    model = create_rscd_model(NUM_CLASSES)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()

def get_rscd_transform():
    # We do our own resize; here only normalization
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def main(input_vid, output_vid, rscd_weights, yolo_weights, no_cuda):
    device = torch.device("cuda" if (not no_cuda and torch.cuda.is_available()) else "cpu")

    # Load RSCD model
    rscd = load_rscd_model(rscd_weights, device)
    rtrans = get_rscd_transform()

    # Load YOLOv10s
    yolo = YOLO(yolo_weights)       # best.pt
    yolo.to(device)
    yolo.conf = 0.01
    yolo.iou  = 0.4

    cap = cv2.VideoCapture(input_vid)
    assert cap.isOpened(), f"Cannot open {input_vid}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_vid, fourcc, fps, (W,H))
    assert out.isOpened(), "Cannot open VideoWriter"

    # trapezoid fractions
    bottom_frac = 0.475
    top_frac    = 0.1
    top_y_frac  = 0.7
    mid_x_frac  = 0.385

    fps_buf = []
    idx = 0
    t_tot = 0.0
    flag = None
    objdetect_dir = 'tallinn_objdetect_frames_y12s'
    # create directory if it doesn't exist
    if not os.path.exists(objdetect_dir):
        os.makedirs(objdetect_dir)
    while True:
        flag = False
        ret, frame = cap.read()
        if not ret: break
        idx +=1
        if idx<500: continue  # skip warmup

        # compute trapezoid corners
        bw = bottom_frac*W
        tw = top_frac*W
        by = H
        ty = int(H*top_y_frac)
        mx = W*mid_x_frac

        bl = (mx-bw/2, by)
        br = (mx+bw/2, by)
        tl = (mx-tw/2, ty)
        tr = (mx+tw/2, ty)

        # --- RSCD pipeline ---
        src = np.array([tl,tr,br,bl],dtype=np.float32)
        # warp_wh = (512,512)
        # dst = np.array([[0,0],warp_wh,[warp_wh[0],warp_wh[1]],[0,warp_wh[1]]],dtype=np.float32)

        # M = cv2.getPerspectiveTransform(src,dst)
        # bird = cv2.warpPerspective(frame, M, warp_wh)
        bird_x1 = int(mx-tw/2); bird_x2 = int(mx+tw/2)
        bird_y1 = int(ty); bird_y2 = int(ty+tw)
                
        bird = frame[bird_y1: bird_y2, bird_x1: bird_x2]
        #Draw a rectangle around the bird's eye view
        cv2.rectangle(frame, (bird_x1, bird_y1), (bird_x2, bird_y2), (0, 100, 0), 2)
        print(f"bird shape: {bird.shape}")
        if idx in [15000, 17500, 20000]:
            cv2.imwrite(f"inference_{idx}.png", bird)
        crop224 = cv2.resize(bird,(224,224),interpolation=cv2.INTER_LINEAR)

        # RSCD inference
        pil = Image.fromarray(cv2.cvtColor(crop224,cv2.COLOR_BGR2RGB))
        inp = rtrans(pil).unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            logits = rscd(inp)
            probs  = F.softmax(logits,1)       
            conf,cls=probs.max(1) #
            # conf, cls = logits.max(1) # logits
        
        # --- YOLO pipeline on axis-aligned ROI ---
        x1 = int((mx-bw/2)-20); x2 = int((mx+bw/2)+20)
        y1 = ty;           y2 = by
        roi = frame[y1:y2, x1:x2] #this is axis-aligned where 
        #add better contrast to roi
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        #Draw a rectangle around the ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 0, 0), 2)
        rH,rW = roi.shape[:2]
        roi512 = cv2.resize(roi,(512,512),interpolation=cv2.INTER_LINEAR)

        # run YOLO
        # y0 = time.time()
        results = yolo(roi512)[0]
        # boxes in 512×512 coords
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs   = results.boxes.conf.cpu().numpy()

        # map YOLO boxes back
        sx,sy = rW/512, rH/512
        for (bx1,by1,bx2,by2), cid, cval in zip(boxes,classes,confs):
            xA = int(bx1*sx + x1)
            yA = int(by1*sy + y1)
            xB = int(bx2*sx + x1)
            yB = int(by2*sy + y1)
            lbl= yolo.names[cid]
            if lbl in ['D00', 'D10', 'D20', 'D40', 'D50']:
                cv2.rectangle(frame,(xA,yA),(xB,yB),(0,0,200),2)
                cv2.putText(frame,f"{lbl} {cval:.2f}",(xA,yA-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                flag = True
        dt = time.time()-t0
        t_tot += dt
        # overlay RSCD result + FPS
        fps_buf.append(1/dt if dt>0 else 0)
        if len(fps_buf)>5: fps_buf.pop(0)
        fps_val = sum(fps_buf)/len(fps_buf)

        txt1 = f"{RSCD_CLASSES[cls.item()]} ({conf.item()*100:4.1f}%)"
        txt2 = f"inference FPS: {fps_val:.2f}"
        txt3 = "NVIDIA GTX1650"
        cv2.putText(frame,txt1,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)
        cv2.putText(frame,txt2,(10,80),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)
        cv2.putText(frame,txt3,(10,120),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)
        
        # draw trapezoid
        cv2.polylines(frame,[np.int32(src)],True,(0,200,0),2)
        if flag:
            
            cv2.imwrite(f"{objdetect_dir}/yolo_{idx}.png", frame)
        out.write(frame)

        if idx%100==0:
            print(f"Processed {idx} frames…")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_vid}")
    print(f"Avg inference FPS: {idx/t_tot:.2f}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',  type=str, default='video.h264')
    p.add_argument('--output', type=str, default='output.mp4')
    p.add_argument('--rscd',   type=str, default='mobilenet_v3_large_weights.pth')
    p.add_argument('--yolo',   type=str, default='best.pt')
    p.add_argument('--no-cuda',action='store_true')
    args = p.parse_args()
    main(args.input, args.output, args.rscd, args.yolo, args.no_cuda)
