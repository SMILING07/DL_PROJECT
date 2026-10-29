"""Fine-tune ViT-B/32 for 4-class brain MRI classification.

Paper: Asiri et al., Sensors 2023, 23, 7913 (FT-ViT, "VIT.pdf").

Paper recipe implemented below (see CONFIG block):
    optimizer  : Adam, learning rate 1e-4
    epochs     : 10, with early stopping on validation accuracy
    head       : Flatten -> BN -> Dense(11) -> BN -> Dense(4)   [Table 2]
    augmentation: random rotations, flips, zoom, brightness
                 (Keras ImageDataGenerator in the paper)
    input      : 224x224 RGB + ImageNet normalization (required by the
                 torchvision pretrained backbone)

Run:            python VITB32_train.py
Sanity check:   set SMOKE_TEST = False  (1 epoch on 32 samples)
Evaluate after: python VITB32_evaluate.py --split Testing.pt
"""
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.transforms import v2
from tqdm.auto import tqdm


# ============================ MODEL (FT-ViT paper) ==========================
class PaperHead(nn.Module):
    """FT-ViT classification head: Flatten -> BN -> Dense(11) -> BN -> Dense(4).

    Paper Table 2 (ViT-b32 column): params match exactly. ReLU after the
    11-unit dense layer is assumed (a fully linear stack would collapse).
    """

    def __init__(self, in_features: int = 768, bottleneck: int = 11, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, bottleneck),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(bottleneck),
            nn.Linear(bottleneck, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_vit_b32(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ViT-B/32 with the ImageNet head replaced by the paper's task head."""
    weights = ViT_B_32_Weights.DEFAULT if pretrained else None
    model = vit_b_32(weights=weights)
    model.heads = PaperHead(in_features=768, bottleneck=11, num_classes=num_classes)
    return model

# ============================ CONFIG (paper values) =========================
SEED = 42
EPOCHS = 10               # paper: "training for a total of 10 epochs"
LR = 1e-4                 # paper: "a learning rate of 1e-4 and the Adam optimizer"
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 3   # paper: "early stopping is crucial to prevent overfitting"
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]
WEIGHT_DECAY = 0.0        # paper uses plain Adam without weight decay
SMOKE_TEST = False        # True -> 1 epoch on 32 samples (quick sanity run)
CKPT_DIR = "vit_checkpoint"   # all checkpoints and plots are written here
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats, as used for the pretrained ViT
STD = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================== DATA ========================================
class BrainTumorDataset(Dataset):
    """Holds the [N,3,224,224] float tensors from Training.pt / Testing.pt.

    Normalization is applied on demand so we never duplicate the full
    image tensor in RAM. Augmentation only runs on the training split.
    """

    def __init__(self, images, labels, train: bool):
        self.images = images
        self.labels = labels
        if train:
            # paper: random rotations, flips, zoom, brightness adjustments
            self.transform = v2.Compose([
                v2.RandomResizedCrop(224, scale=(0.8, 1.0)),  # zoom
                v2.RandomHorizontalFlip(),                    # flips (50%)
                v2.RandomVerticalFlip(),
                v2.RandomRotation(15),                        # rotations
                v2.ColorJitter(brightness=0.3),               # brightness
                v2.Normalize(MEAN, STD),
            ])
        else:
            self.transform = v2.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx].float()), self.labels[idx]


def build_loaders():
    """Load the .pt splits and return train / val / test DataLoaders."""
    train_blob = torch.load("Training.pt", weights_only=True, map_location="cpu")
    x_train, y_train = train_blob["images"], train_blob["labels"].long()
    del train_blob
    test_blob = torch.load("Testing.pt", weights_only=True, map_location="cpu")
    x_test, y_test = test_blob["images"], test_blob["labels"].long()
    del test_blob

    # 80/20 stratified validation split of the training set
    idx = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.20, random_state=SEED, stratify=y_train.numpy()
    )
    if SMOKE_TEST:  # tiny subset for the sanity run
        train_idx, val_idx = train_idx[:32], val_idx[:32]

    train_loader = DataLoader(
        Subset(BrainTumorDataset(x_train, y_train, train=True), train_idx),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(BrainTumorDataset(x_train, y_train, train=False), val_idx),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda",
    )
    test_loader = DataLoader(
        BrainTumorDataset(x_test, y_test, train=False),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda",
    )
    print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(y_test)}")
    return train_loader, val_loader, test_loader


# ========================= TRAIN / EVAL LOOPS ===============================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = correct = total = 0
    for images, labels in tqdm(loader, desc="Training", unit="batch", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.size(0)
        running_loss += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        total += n
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = correct = total = 0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        pred = logits.argmax(1)
        n = labels.size(0)
        running_loss += loss.item() * n
        correct += (pred == labels).sum().item()
        total += n
        preds_all.extend(pred.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return running_loss / total, correct / total, preds_all, labels_all


def fit(model, train_loader, val_loader):
    """Train EPOCHS epochs, save the best-val checkpoint, early-stop on plateau.

    Returns (history, best_state, best_val_acc, stop_epoch). stop_epoch is the
    epoch at which early stopping triggered, else None. Best checkpoint is
    written to vit_checkpoint/vit_b32_best.pth on every val-acc improvement.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state, best_acc, epochs_no_improve = None, -1.0, 0
    stop_epoch = None

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Fine-tuning", unit="epoch"):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch}/{EPOCHS} | train {tr_loss:.4f}/{tr_acc*100:.2f}% "
              f"| val {va_loss:.4f}/{va_acc*100:.2f}%")

        if va_acc > best_acc:  # checkpoint on best validation accuracy
            best_acc, best_state, epochs_no_improve = va_acc, copy.deepcopy(model.state_dict()), 0
            torch.save(
                {"epoch": epoch, "model_state_dict": best_state,
                 "val_acc": va_acc, "val_loss": va_loss, "train_loss": tr_loss},
                f"{CKPT_DIR}/vit_b32_best.pth",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val improvement "
                      f"for {EARLY_STOP_PATIENCE} epochs)")
                stop_epoch = epoch
                break

    return history, best_state, best_acc, stop_epoch


def plot_results(history, stop_epoch=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
    # mark where early stopping halted training (paper: early stopping
    # prevents overfitting when val accuracy plateaus)
    if stop_epoch is not None:
        for ax in (ax1, ax2):
            ax.axvline(stop_epoch, linestyle="--", color="gray", linewidth=1)
        ax2.annotate(f"early stop\n(epoch {stop_epoch})", xy=(stop_epoch, 0.02),
                     xycoords=("data", "axes fraction"), fontsize=7,
                     ha="center", color="gray")
    ax1.legend(); ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{CKPT_DIR}/vit_b32_curves.png", dpi=150)
    plt.show()


def main():
    print("Device:", DEVICE)
    Path(CKPT_DIR).mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = build_loaders()

    model = build_vit_b32(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history, best_state, best_val, stop_epoch = fit(model, train_loader, val_loader)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val * 100:.2f}%")

    # ---- final test report ----
    _, test_acc, preds, labels = evaluate(model, test_loader, nn.CrossEntropyLoss())
    print("\n========== TEST RESULTS ==========")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(classification_report(labels, preds, labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES, zero_division=0, digits=4))
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
    plt.title("ViT-B/32 Brain Tumor Classification")
    plt.tight_layout()
    plt.savefig(f"{CKPT_DIR}/vit_b32_confusion_matrix.png", dpi=150)
    plt.show()

    plot_results(history, stop_epoch)
    print(f"\nBest checkpoint saved to {CKPT_DIR}/vit_b32_best.pth")
    print("Evaluate with: python VITB32_evaluate.py --split Testing.pt")


if __name__ == "__main__":
    main()
