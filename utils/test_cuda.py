import ultralytics
import torch
import torchvision


print("Torch version: ", torch.__version__)
print("Ultralytics version: ", ultralytics.__version__)
print("CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("CUDNN version: ", torch.backends.cudnn.version())

device = 'cuda'
boxes = torch.tensor([[0., 1., 2., 3.]]).to(device)
scores = torch.randn(1).to(device)
iou_thresholds = 0.5

print(torchvision.ops.nms(boxes, scores, iou_thresholds))