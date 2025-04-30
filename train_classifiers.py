import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.optim
import torch.optim.lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from thop import profile


# Device configuration: use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Hyperparameters
batch_size = 128               # Batch size is 128
lambda_center = 0.02           # Weight for center loss  Zhao et al. was 2.0
lr_ce_loss = 0.001             # Learning rate for cross-entropy loss
lr_center_loss = 0.0005        # Learning rate for center loss
label_smoothing_value = 0.1    # this gives 90% weight to correct class, 10% distributed to others
num_workers = 0                # Number of workers for DataLoader
num_epochs = 6                 # Number of epochs
dropout_rate = 0.2            # Dropout rate for the model
weight_decay = 1e-5           # Weight decay for optimizer

# ImageNet normalization values
# These values are used to normalize the input images for models pretrained on ImageNet.
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]



# Dataset & Transforms
class RoadSurfaceDataset(Dataset):
    def __init__(self, directory, transform=None, class_to_idx=None):
        """
        Custom Dataset for road surface images.
        - directory: path to the folder containing images.
        - transform: torchvision transforms to apply.
        - class_to_idx: optional dict mapping label strings to indices.
        """
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = class_to_idx

        # Walk through directory (and subdirectories)
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(root, file)
                    # Example filename: "20220429231742925-wet-gravel.jpg" = label "wet-gravel"
                    parts = file.split('-')
                    label_str = "-".join(parts[1:]).split('.')[0]
                    self.images.append(full_path)
                    self.labels.append(label_str)
        
        # If mapping is not provided, build it from unique labels here.
        if self.class_to_idx is None:
            unique_labels = sorted(set(self.labels))
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert labels to numeric indices.
        self.labels = [self.class_to_idx[l] for l in self.labels]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    

def get_transforms(img_size):
        # ImageNet normalization    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, val_transform


# Loss Functions: CE + Center Loss
import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Source: Kaiyang Zhou Center Loss implementation: https://github.com/KaiyangZhou/pytorch-center-loss
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    

def create_classifier(feat_dim, num_classes, dropout_rate=0.2, use_batchnorm=True):
    layers = []
    if dropout_rate > 0:
        layers.append(nn.Dropout(p=dropout_rate))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(feat_dim))
    layers.append(nn.Linear(feat_dim, num_classes))
    return nn.Sequential(*layers)


def create_model(model_name, num_classes):
    if model_name == "mnasnet1_3":
        base_model = models.mnasnet1_3(pretrained=True)
        in_features = base_model.classifier[1].in_features  # e.g., 1280
        # Replace the classifier with a new Linear layer.
        base_model.classifier[1] = nn.Linear(in_features, num_classes)
        
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=lr_ce_loss,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=6, eta_min=1e-6
        )
        # For MNASNet, the feature extractor is stored in base_model.layers.
        backbone = base_model.layers
        
    elif model_name == "shufflenet_v2_x1_5":
        base_model = models.shufflenet_v2_x1_5(pretrained=True)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
        
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=lr_ce_loss,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=6, eta_min=1e-6
        )
        backbone = nn.Sequential(*list(base_model.children())[:-1])
        
    elif model_name == "regnet_y_400mf":
        base_model = models.regnet_y_400mf(pretrained=True)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
        
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=lr_ce_loss,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=6, eta_min=1e-6
        )
        backbone = nn.Sequential(*list(base_model.children())[:-1])
        
    elif model_name == "mobilenet_v3_large":
        base_model = models.mobilenet_v3_large(pretrained=True)
        # In MobileNetV3_Large, the backbone (base_model.features) outputs 960 channels.
        in_features = 960  # fixed number for MobileNetV3_Large's features.
        
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=lr_ce_loss,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=6, eta_min=1e-6
        )
        backbone = base_model.features
        
    elif model_name == "efficientnet_b0":
        base_model = models.efficientnet_b0(pretrained=True)
        in_features = base_model.classifier[1].in_features 
        base_model.classifier[1] = nn.Linear(in_features, num_classes)
        
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=lr_ce_loss,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=6, eta_min=1e-6
        )
        backbone = base_model.features
        
    else:
        raise ValueError(f"Model {model_name} not supported!")
    
    # Create a custom classifier using our helper function.
    classifier = create_classifier(in_features, num_classes, dropout_rate=dropout_rate, use_batchnorm=True)
    
    # Wrap the backbone and classifier into a new module.
    class ModelWithEmbedding(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
        def forward(self, x):
            features = self.backbone(x)
            # If features have spatial dimensions, apply adaptive pooling to get (batch, channels, 1, 1).
            if features.dim() == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1))
            # Flatten to (batch, channels)
            features = torch.flatten(features, start_dim=1)
            logits = self.classifier(features)
            return logits, features

    wrapped_model = ModelWithEmbedding(backbone, classifier)
    return wrapped_model.to(device), lr_ce_loss, optimizer, scheduler, in_features


# Metrics and Inference Timing
def compute_metrics(all_preds, all_labels, num_classes):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)
    accuracy = (all_preds == all_labels).sum() / total

    precision_list, recall_list = [], []
    for cls in range(num_classes):
        TP = np.logical_and(all_preds == cls, all_labels == cls).sum()
        FP = np.logical_and(all_preds == cls, all_labels != cls).sum()
        FN = np.logical_and(all_preds != cls, all_labels == cls).sum()
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
    precision_macro = np.mean(precision_list)
    recall_macro = np.mean(recall_list)
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro) if (precision_macro+recall_macro) > 0 else 0.0
    return accuracy, precision_macro, recall_macro, f1_macro

def evaluate(model, data_loader, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc, precision_macro, recall_macro, f1_macro = compute_metrics(all_preds, all_labels, num_classes)
    return acc, precision_macro, recall_macro, f1_macro

def measure_inference_time(model, data_loader):
    model.eval()
    total_images = 0
    total_time = 0.0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            start = time.time()
            _ = model(images)
            end = time.time()
            total_time += (end - start)
            total_images += images.size(0)
    avg_time = total_time / total_images if total_images > 0 else 0
    return avg_time * 1000  # milliseconds per image



# Training Loop
def train_one_epoch(model, criterion_ce, center_loss_fn, optimizer, data_loader, epoch, lambda_center=0.0):    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
        # Move images and labels to device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        # Forward pass
        logits, features = model(images)
        
        # Compute losses        
        loss_ce = criterion_ce(logits, labels)
        loss_center = center_loss_fn(features, labels)
        loss = loss_ce + lambda_center * loss_center
        
        # Backpropagation
        loss.backward()
        
        for param in center_loss_fn.parameters():
            param.grad.data *= (1./lambda_center)
        
        optimizer.step()
        
        # Update running statistics
        running_loss += loss.item() * images.size(0)
        # For accuracy, check top-1 predictions
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Train Acc = {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy


#####################################
# Main Loop: Train, Validate, Test for Each Model
#####################################

# List of models to test.
models_list = ["mnasnet1_3", "shufflenet_v2_x1_5", "regnet_y_400mf", "mobilenet_v3_large", "efficientnet_b0"]


# Dataset directories – update this to point to your RSCD dataset.
data_dir = "RSCD_dataset" 
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "vali_20k")
test_dir  = os.path.join(data_dir, "test_50k")

# Root folder for saving per-model results.
results_root = "results"
os.makedirs(results_root, exist_ok=True)

# Summary for all models.
summary_metrics = []

for model_name in models_list:
    print(f"\n==== Training {model_name} ====")
    input_size = 224
    train_transform, val_transform = get_transforms(input_size)
    
    # Create datasets.
    train_dataset = RoadSurfaceDataset(train_dir, transform=train_transform)
    val_dataset = RoadSurfaceDataset(val_dir, transform=val_transform,class_to_idx=train_dataset.class_to_idx)
    test_dataset = RoadSurfaceDataset(test_dir, transform=val_transform, class_to_idx=train_dataset.class_to_idx)
    num_classes = len(train_dataset.class_to_idx)
    
    # train_dataset = Subset(train_dataset, range(int(len(train_dataset) * 0.01)))
    # val_dataset = Subset(val_dataset, range(int(len(val_dataset) * 0.01)))
    # test_dataset = Subset(test_dataset, range(int(len(test_dataset) * 0.01)))
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Create separate folder for this model.
    model_folder = os.path.join(results_root, model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    # Create model
    model, lr, optimizer, scheduler, feat_dim = create_model(model_name, num_classes)
    print(f"Model {model_name} created with feature dimension: {feat_dim}")
    
    #Instantiate loss functions.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_value).to(device)
    center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True).to(device)
    
    optimizer.add_param_group({"params": center_loss_fn.parameters(), "lr": lr_center_loss})
    
     # Logging lists.
    epochs_log = []
    train_loss_log = []
    train_acc_log = []
    val_acc_log = []
    val_precision_log = []
    val_recall_log = []
    val_f1_log = []
    
    
    # Training loop.
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, center_loss_fn, optimizer, train_loader, epoch, lambda_center)
        accuracy, precision, recall, f1 = evaluate(model, val_loader, num_classes)
        print(f"[Val] Epoch {epoch+1}: Acc={accuracy:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        epochs_log.append(epoch+1)
        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)
        val_acc_log.append(accuracy)
        val_precision_log.append(precision)
        val_recall_log.append(recall)
        val_f1_log.append(f1)
        scheduler.step()  # Update learning rate
        
    # Save model weights.
    weights_path = os.path.join(model_folder, f"{model_name}_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved {model_name} weights to {weights_path}")
    
    # Log metrics to CSV.
    df_metrics = pd.DataFrame({
        "Epoch": epochs_log,
        "Train_Loss": train_loss_log,
        "Train_Acc": train_acc_log,
        "Val_Acc": val_acc_log,
        "Val_Precision": val_precision_log,
        "Val_Recall": val_recall_log,
        "Val_F1": val_f1_log
    })
    csv_path = os.path.join(model_folder, f"{model_name}_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved CSV metrics to {csv_path}")
    
    # Plot training curves and save as JPG.
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_log, train_loss_log, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(epochs_log, train_acc_log, marker="o", label="Train Acc")
    plt.plot(epochs_log, val_acc_log, marker="o", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(model_folder, f"{model_name}_training_plots.jpg")
    plt.tight_layout()
    plt.savefig(plot_path, format="jpg")
    plt.close()
    print(f"Saved training plots to {plot_path}")
    
    # Measure model complexity: GFLOPs and parameter count.
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    mac, _ = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = 2 * (mac / 1e9)  # Convert to GFLOPs    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {model_name} GFLOPs: {gflops:.2f} Parameters: {param_count/1e6:.2f}M")
    
    # Test evaluation.
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, num_classes)
    avg_inf_time = measure_inference_time(model, test_loader)
    print(f"[Test] {model_name}: Acc={test_acc:.4f}, Prec={test_prec:.4f}, Recall={test_rec:.4f}, F1={test_f1:.4f}")
    print(f"Avg Inference Time per image: {avg_inf_time:.2f} ms")
    
    #Select 5 random samples from the test set for visualization.
    # sample_indices = random.sample(range(len(test_dataset.dataset)), 5)
    # idx_to_class = {v: k for k, v in test_dataset.dataset.class_to_idx.items()}
    sample_indices = random.sample(range(len(test_dataset)), 5)
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
            
    # Prepare to display images with matplotlib.
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), facecolor='white')
    fig.subplots_adjust(wspace=0.3)

    # Set the model in evaluation mode.
    model.eval()
    softmax = nn.Softmax(dim=1)

    for i, idx in enumerate(sample_indices):
        # image_tensor, true_label = test_dataset.dataset[idx]
        image_tensor, true_label = test_dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(input_tensor)
            probs = softmax(logits)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_label = idx_to_class[pred_idx.item()]
            conf_percent = conf.item() * 100

        # Convert the image tensor to a format that matplotlib can display.
        # First, unnormalize the tensor.
        inv_norm = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
        disp_tensor = inv_norm(image_tensor).clamp(0, 1)
        # Convert to numpy (transpose to H, W, C).
        img_np = disp_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Plot in the corresponding subplot.
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f"{pred_label}\n({conf_percent:.1f}%)", fontsize=14)

    # Save the resulting plot as a JPG.
    plot_save_path = os.path.join(model_folder, f"{model_name}_test_predictions_matplotlib.jpg")
    plt.savefig(plot_save_path, format="jpg", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved test predictions plot to {plot_save_path}")
    
    # Append final metrics for summary.
    summary_metrics.append({
        "Model": model_name,
        "Test_Acc": test_acc,
        "Test_Precision": test_prec,
        "Test_Recall": test_rec,
        "Test_F1": test_f1,
        "Param_Count": param_count,
        "GFLOPs": gflops,
        "Avg_Inference_Time_ms": avg_inf_time
    })
    

# Save overall summary CSV.
summary_df = pd.DataFrame(summary_metrics)
summary_csv_path = os.path.join(results_root, "models_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Saved overall summary metrics to {summary_csv_path}")


for model_name in models_list:

    # Replace with the path to your metrics CSV file.
    csv_file_path = f"results/{model_name}/{model_name}_metrics.csv"

    # Read the CSV file.
    df = pd.read_csv(csv_file_path)

    # Create a figure with 5 subplots (one per metric).
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True)

    # Plot 1: Train Loss
    axes[0].plot(df["Epoch"], df["Train_Loss"], marker="o", color="blue", label="Train Loss")
    axes[0].set_title("Train Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")
    # axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Validation Accuracy
    axes[1].plot(df["Epoch"], df["Val_Acc"], marker="o", color="green", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    # axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Validation Precision
    axes[2].plot(df["Epoch"], df["Val_Precision"], marker="o", color="red", label="Val Precision")
    axes[2].set_title("Validation Precision")
    axes[2].set_ylabel("Precision")
    axes[2].set_xlabel("Epoch")
    # axes[2].legend()
    axes[2].grid(True)

    # Plot 4: Validation Recall
    axes[3].plot(df["Epoch"], df["Val_Recall"], marker="o", color="orange", label="Val Recall")
    axes[3].set_title("Validation Recall")
    axes[3].set_ylabel("Recall")
    axes[3].set_xlabel("Epoch")
    # axes[3].legend()
    axes[3].grid(True)

    # Plot 5: Validation F1 Score
    axes[4].plot(df["Epoch"], df["Val_F1"], marker="o", color="purple", label="Val F1")
    axes[4].set_title("Validation F1 Score")
    axes[4].set_ylabel("F1 Score")
    axes[4].set_xlabel("Epoch")
    # axes[4].legend()
    axes[4].grid(True)

    # Adjust layout and save the figure as JPG.
    plt.tight_layout()
    plt.savefig(f"results/{model_name}/{model_name}_metrics_plot.jpg", format="jpg")
    plt.show()
    
    
# Read the summary CSV file.
csv_file = "results_megarun2/models_summary.csv"  # Update this path if needed.
df = pd.read_csv(csv_file)

# Ensure the CSV has the required columns: 
# "Model", "Test_Acc", "Param_Count", "GFLOPs", "Avg_Inference_Time_ms"

# Create a consistent color mapping for each model.
# Extract unique models (preserving order if needed)
unique_models = df["Model"].unique()

# You can choose your colors; here we use a fixed list (make sure you have at least as many colors as models).
default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
# Create a dictionary mapping each model to a color.
color_map = {model: default_colors[i % len(default_colors)] for i, model in enumerate(unique_models)}

# For each row in the DataFrame, get the corresponding color.
bar_colors = [color_map[model] for model in df["Model"]]

# Create a 1x4 subplots figure.
fig, axes = plt.subplots(1, 4, figsize=(12, 4), facecolor='white')
fig.suptitle("Model Summary", fontsize=16)

# Bar chart for Test Accuracy.
axes[0].bar(df["Model"], df["Test_Acc"], color=bar_colors)
axes[0].set_title("Test Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)  # Accuracy between 0 and 1.

# Bar chart for Parameter Count.
axes[1].bar(df["Model"], df["Param_Count"], color=bar_colors)
axes[1].set_title("Parameter Count")
axes[1].set_ylabel("Parameters")

# Bar chart for GFLOPs.
axes[2].bar(df["Model"], df["GFLOPs"], color=bar_colors)
axes[2].set_title("GFLOPs")
axes[2].set_ylabel("GFLOPs")

# Bar chart for Avg Inference Time (ms).
axes[3].bar(df["Model"], df["Avg_Inference_Time_ms"], color=bar_colors)
axes[3].set_title("Avg Inference Time (ms)")
axes[3].set_ylabel("Time (ms)")

# Set x-axis labels with rotation and grid in all subplots.
for ax in axes:
    ax.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = "results_megarun2/model_summary.jpg"
plt.savefig(output_path, format="jpg")
plt.show()

print(f"Saved summary plot to {output_path}")