import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import os
import copy
from sklearn.metrics import precision_score, recall_score, f1_score

# ========== Data Preprocessing ==========
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

train_path = r"/home/sfmt/Downloads/6483/datasets/datasets/train"
val_path = r"/home/sfmt/Downloads/6483/datasets/datasets/val"

full_train_dataset = datasets.ImageFolder(train_path, data_transforms['train'])
full_val_dataset = datasets.ImageFolder(val_path, data_transforms['val'])

np.random.seed(42)
train_indices = np.random.choice(len(full_train_dataset), 2000, replace=False)
val_indices = np.random.choice(len(full_val_dataset), 500, replace=False)

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)
print(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== Model Building ==========
vgg_model = models.vgg16(pretrained=True)

for param in vgg_model.features.parameters():
    param.requires_grad = False

vgg_model.classifier[6] = nn.Linear(4096, 2)
vgg_model = vgg_model.to(device)

# ========== Loss function & Optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.classifier.parameters(), lr=0.00001)

# ========== Learning rate ==========
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# ========== training set ==========

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=60, scheduler=None):
    train_losses = []
    val_losses = []

    # 初始化 Early Stopping 参数
    patience = 5
    delta = 0.00001
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = "vgg_catdog_best.pth"

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        model.train()
        running_loss = 0.0
        correct = 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix({
                'loss': loss.item(),
                'lr': f"{current_lr:.6f}"
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if scheduler:
            scheduler.step()

        tqdm.write(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, LR: {current_lr:.6f}")

        model.eval()
        val_loss = 0.0
        val_correct = 0

        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        tqdm.write(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

# ========== test set ==========

def predict_on_test(model, test_path, device, output_csv="test_predictions.csv"):
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class TestDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_paths = sorted([
                os.path.join(image_dir, fname)
                for fname in os.listdir(image_dir)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
            ], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            filename = os.path.splitext(os.path.basename(img_path))[0]
            if self.transform:
                image = self.transform(image)
            return image, filename

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    results = []

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
            for fname, pred in zip(filenames, predicted):
                results.append({"filename": fname, "label": int(pred)})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Prediction saved to {output_csv}")

# ========== F score ==========
def evaluate_predictions(pred_csv_path, groundtruth_csv_path):
    pred_df = pd.read_csv(pred_csv_path)
    gt_df = pd.read_csv(groundtruth_csv_path)

    pred_df = pred_df.sort_values("filename").reset_index(drop=True)
    gt_df = gt_df.sort_values("filename").reset_index(drop=True)

    y_pred = pred_df["label"].astype(int).tolist()
    y_true = gt_df["label"].astype(int).tolist()

    TP = TN = FP = FN = 0
    error_files = []

    for pred_label, true_label, fname in zip(y_pred, y_true, pred_df["filename"]):
        if pred_label == 1 and true_label == 1:
            TP += 1
        elif pred_label == 0 and true_label == 0:
            TN += 1
        elif pred_label == 1 and true_label == 0:
            FP += 1
            error_files.append(fname)
        elif pred_label == 0 and true_label == 1:
            FN += 1
            error_files.append(fname)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Prediction result：")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("wrong prediction indices:")
    for fname in error_files:
        print(f" - {fname}")

def evaluate_on_loader(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    print("Validation result：")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")



# ====================
# train_model(vgg_model, train_loader, val_loader, criterion, optimizer, device, num_epochs=60, scheduler = None)
#
# vgg_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# vgg_model.classifier[6] = nn.Linear(4096, 2)
# vgg_model.load_state_dict(torch.load("vgg_catdog_final.pth", map_location=device))
# vgg_model = vgg_model.to(device)
#
# evaluate_on_loader(vgg_model, val_loader, device)
#
# test_path = r"/home/sfmt/Downloads/6483/datasets/datasets/test"
# predict_on_test(vgg_model, test_path, device, output_csv="test_predictions.csv")
evaluate_predictions("test_predictions.csv", "groundtruth.csv")


# evaluate_predictions("WM.csv", "groundtruth.csv")