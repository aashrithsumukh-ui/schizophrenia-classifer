"""
2D Slice-Based Schizophrenia Classifier using ResNet-18 with Transfer Learning
================================================================================
Approach:
  - Load 3D brain MRI volumes (.npy, 128x128x128)
  - Extract middle 40 axial slices (indices 44–84) per volume
  - Convert each 128x128 slice to 3-channel by repeating
  - Fine-tune ImageNet-pretrained ResNet-18 with a binary output head
  - Stratified train/val/test split (70/15/15) by label AND site
  - Weighted cross-entropy loss for class imbalance
  - Early stopping (patience=10) on validation loss
  - Full evaluation: Accuracy, AUC-ROC, Sensitivity, Specificity, Confusion Matrix
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    # Script auto-detects its own directory so paths work regardless of where
    # you launch Python from. Both the CSV and the saved checkpoint land next
    # to the script file.
    _here          = os.path.dirname(os.path.abspath(__file__))
    csv_path       = os.path.join(_here, "harmonized_labels.csv")
    checkpoint     = os.path.join(_here, "best_model.pth")

    slice_axis     = 2                          # axial slices
    slice_start    = 44                         # first slice index (inclusive)
    slice_end      = 84                         # last slice index (exclusive) → 40 slices
    img_size       = 128
    num_classes    = 2
    batch_size     = 64
    # Windows + Python multiprocessing requires num_workers=0 unless the script
    # is protected by  if __name__ == "__main__"  (which it is here).
    # If you see DataLoader worker crashes, set this to 0.
    num_workers    = 0
    lr             = 1e-4
    max_epochs     = 100
    patience       = 10                         # early-stopping patience
    seed           = 42
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
print(f"Using device: {cfg.device}")


# ─────────────────────────────────────────────
# 1. DATA LOADING & SPLITTING
# ─────────────────────────────────────────────

def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load the label CSV and create a stratification key from label + site."""
    df = pd.read_csv(csv_path)
    # Combined stratum for stratified splitting
    df["stratum"] = df["label"].astype(str) + "_" + df["site"].astype(str)
    return df


def split_subjects(df: pd.DataFrame, seed: int):
    """
    70 / 15 / 15 train/val/test split stratified by (label, site).
    Returns three DataFrames indexed by subject.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["stratum"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["stratum"], random_state=seed
    )
    print(f"Subjects  → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────

# ImageNet normalisation applied to each (repeated) slice
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class BrainSliceDataset(Dataset):
    """
    Each MRI volume contributes (slice_end - slice_start) individual 2-D samples.
    Label is the same for all slices from the same subject.
    """

    def __init__(self, subject_df: pd.DataFrame, transform=None,
                 slice_start: int = 44, slice_end: int = 84):
        self.transform   = transform
        self.slice_start = slice_start
        self.slice_end   = slice_end
        self.n_slices    = slice_end - slice_start   # 40

        # Build flat index: list of (filepath, slice_idx, label)
        self.samples = []
        for _, row in subject_df.iterrows():
            for s in range(slice_start, slice_end):
                self.samples.append((row["filepath"], s, int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, slice_idx, label = self.samples[idx]

        # Load 3-D volume and extract one axial slice
        volume = np.load(filepath).astype(np.float32)   # (128, 128, 128)
        slice2d = volume[:, :, slice_idx]               # (128, 128)

        # Replace NaN / Inf that can appear in preprocessed MRI data
        slice2d = np.nan_to_num(slice2d, nan=0.0, posinf=0.0, neginf=0.0)

        # Min-max normalise to [0, 255] uint8 so PIL can handle it
        lo, hi = slice2d.min(), slice2d.max()
        if hi > lo:
            slice2d = (slice2d - lo) / (hi - lo)
        else:
            slice2d = np.zeros_like(slice2d)   # blank slice — all values identical
        slice2d = np.clip(slice2d * 255, 0, 255).astype(np.uint8)

        # Repeat to 3 channels → (128, 128, 3)
        slice_rgb = np.stack([slice2d, slice2d, slice2d], axis=-1)

        if self.transform:
            img = self.transform(slice_rgb)
        else:
            img = torch.from_numpy(slice_rgb.transpose(2, 0, 1)).float() / 255.0

        return img, label


def make_loaders(train_df, val_df, test_df, cfg):
    """Build DataLoaders; use WeightedRandomSampler for the training set."""
    train_ds = BrainSliceDataset(train_df, TRAIN_TRANSFORM, cfg.slice_start, cfg.slice_end)
    val_ds   = BrainSliceDataset(val_df,   EVAL_TRANSFORM,  cfg.slice_start, cfg.slice_end)
    test_ds  = BrainSliceDataset(test_df,  EVAL_TRANSFORM,  cfg.slice_start, cfg.slice_end)

    # Class weights for sampler (compensate for SCHZ/CTRL imbalance at the slice level)
    labels_train = [s[2] for s in train_ds.samples]
    class_counts = np.bincount(labels_train)
    sample_weights = [1.0 / class_counts[l] for l in labels_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    pin = torch.cuda.is_available()   # pin_memory only helps on GPU

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=pin)

    print(f"Slices    → train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_counts


# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────

def build_model(num_classes: int = 2) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet.
    Simple head replacement — full fine-tuning from epoch 1.
    Lower dropout to let gradients flow freely.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features   # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────

def compute_class_weights(class_counts: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights for weighted cross-entropy loss."""
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return running_loss / total, correct / total


def train(model, train_loader, val_loader, cfg, class_weights):
    # Standard weighted cross-entropy — reliable and well-behaved
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential LRs from the start: backbone gets 10x less than head
    backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
    head_params     = list(model.fc.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params,     "lr": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=1e-7
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:3d}/{cfg.max_epochs} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.checkpoint)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

    return history


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    """Return true labels, predicted labels, and positive-class probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # P(SCHZ)
        preds  = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(labels, preds, probs):
    """Compute and print all evaluation metrics."""
    acc  = accuracy_score(labels, preds)
    auc  = roc_auc_score(labels, probs)
    cm   = confusion_matrix(labels, preds)   # rows=actual, cols=predicted

    # TN, FP, FN, TP (binary confusion matrix)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)   # recall for SCHZ class
    specificity = tn / (tn + fp)   # recall for CTRL class

    print("\n" + "="*55)
    print("          TEST SET EVALUATION RESULTS")
    print("="*55)
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}  (SCHZ recall)")
    print(f"  Specificity : {specificity:.4f}  (CTRL recall)")
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                Pred CTRL  Pred SCHZ")
    print(f"  Actual CTRL:    {tn:5d}      {fp:5d}")
    print(f"  Actual SCHZ:    {fn:5d}      {tp:5d}")
    print("="*55)

    return {"accuracy": acc, "auc": auc, "sensitivity": sensitivity,
            "specificity": specificity, "confusion_matrix": cm}


# ─────────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    """Save a 2-panel figure: loss curves and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Validation")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Validation")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load metadata ────────────────────────────────────────
    print("\n[1/5] Loading metadata …")
    df = load_metadata(cfg.csv_path)
    print(f"  Total subjects: {len(df)}  |  SCHZ: {(df.label==1).sum()}  CTRL: {(df.label==0).sum()}")

    # ── Split subjects ───────────────────────────────────────
    print("\n[2/5] Splitting subjects …")
    train_df, val_df, test_df = split_subjects(df, cfg.seed)

    # ── Build data loaders ───────────────────────────────────
    print("\n[3/5] Building datasets & loaders …")
    train_loader, val_loader, test_loader, class_counts = make_loaders(
        train_df, val_df, test_df, cfg
    )

    # ── Build model ──────────────────────────────────────────
    print("\n[4/5] Building model …")
    model = build_model(cfg.num_classes).to(cfg.device)
    class_weights = compute_class_weights(class_counts, cfg.device)
    print(f"  Class weights (CTRL, SCHZ): {class_weights.cpu().numpy()}")

    # ── Train ────────────────────────────────────────────────
    print("\n[5/5] Training …")
    history = train(model, train_loader, val_loader, cfg, class_weights)

    # ── Plot curves ──────────────────────────────────────────
    plot_training_curves(history)

    # ── Load best checkpoint and evaluate ────────────────────
    print(f"\nLoading best checkpoint from '{cfg.checkpoint}' …")
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=cfg.device))

    labels, preds, probs = get_predictions(model, test_loader, cfg.device)
    metrics = print_metrics(labels, preds, probs)


if __name__ == "__main__":
    main()
