"""RViT-FusionNet (ResNet-50 + ViT-B/16 + Local Cross-Attention) for
4-class brain MRI classification.

Paper: Islam et al., "RViT-FusionNet", Neural Computing & Applications
2026, 38:518  ("RVIT.pdf").

Paper recipe implemented below (see CONFIG block):
    batch      : 16 across train and validation
    epochs     : 50, cosine-annealed learning rates
    optimizer  : AdamW with differential learning rates
                 fusion/classifier 1e-3 | LCA 5e-4 | backbones 1e-5
    augment    : RandomResizedCrop(0.8-1.0), h-flip, v-flip, rotation 15deg,
                 color jitter +-30%, ImageNet normalization
    unfreezing : progressive (Sec. 3.5.4) -- backbones frozen first;
                 deeper ResNet layers at epoch E1; ViT at epoch E2
                 (paper names E1/E2 but gives no values -> chosen below)
    loss       : cross entropy + 0.2 * stochastic domain-discriminator BCE
                 (Sec. 3.4.4: randomized binary labels each cycle,
                 standard backprop, no gradient reversal)
    checkpoints: best validation accuracy (Sec. 3.5.6)

The model definition (RViTFusionNet) lives in this file and is imported by
RVIT_evaluate.py.

Run:            python RVIT_FUSIONNET_train.py
Sanity check:   set SMOKE_TEST = False  (1 epoch on 32 samples)
Evaluate after: python RVIT_evaluate.py --split Testing.pt
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
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.transforms import v2
from tqdm.auto import tqdm


# ============================ MODEL (RViT-FusionNet) ========================
class ResNetFeatures(nn.Module):
    """Final ResNet50 spatial map [B,2048,7,7] and pooled [B,2048]."""
    def __init__(self):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(base.conv1,base.bn1,base.relu,base.maxpool)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
    def forward(self, x):
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        return z, z.mean(dim=(2,3))

class ViTPatchFeatures(nn.Module):
    """torchvision ViT-B/16 encoder, excluding CLS from mean-pooled patch features."""
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()
    def forward(self,x):
        vit = self.vit
        z = vit._process_input(x) # [B,196,768]
        cls = vit.class_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls,z),dim=1)
        z = vit.encoder(z) # includes positional embeddings, dropout, encoder and final LN
        patch_tokens = z[:,1:,:]
        return patch_tokens, patch_tokens.mean(dim=1)

class LocalCrossAttention(nn.Module):
    def __init__(self, embed_dim=768, heads=8):
        super().__init__()
        self.resnet_projection = nn.Conv2d(2048,embed_dim,kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self,resnet_map,vit_tokens):
        queries = self.resnet_projection(resnet_map).flatten(2).transpose(1,2)
        attended,_ = self.attn(queries,vit_tokens,vit_tokens,need_weights=False)
        attended = self.norm(queries + attended)
        return attended.mean(dim=1)

class GatedFeatureFusion(nn.Module):
    def __init__(self,dim=1024):
        super().__init__()
        self.res_proj = nn.Linear(2048,dim)
        self.vit_proj = nn.Linear(768,dim)
        self.local_proj = nn.Linear(768,dim)
        self.gate = nn.Sequential(nn.Linear(dim*3,256),nn.ReLU(),nn.Linear(256,3))
    def forward(self,res_vec,vit_vec,local_vec):
        features = [self.res_proj(res_vec),self.vit_proj(vit_vec),self.local_proj(local_vec)]
        weights = self.gate(torch.cat(features,dim=1)).softmax(dim=1)
        fused = sum(weights[:,i:i+1]*features[i] for i in range(3))
        return fused,weights

class PatchDiscriminator(nn.Module):
    """Paper's 2048->512->256->1 patch logits, without gradient reversal."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2048,512,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256,1,1)
        )
    def forward(self,feature_map):
        return self.net(feature_map)

class RViTFusionNet(nn.Module):
    def __init__(self,num_classes=4):
        super().__init__()
        self.resnet = ResNetFeatures()
        self.vit = ViTPatchFeatures()
        self.lca = LocalCrossAttention()
        self.fusion = GatedFeatureFusion()
        self.classifier = nn.Sequential(
            nn.Linear(1024,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,num_classes)
        )
        self.discriminator = PatchDiscriminator()
    def forward(self,x,return_domain=False):
        res_map,res_vec = self.resnet(x)
        tokens,vit_vec = self.vit(x)
        local_vec = self.lca(res_map,tokens)
        fused,gate = self.fusion(res_vec,vit_vec,local_vec)
        logits = self.classifier(fused)
        if return_domain:
            return logits,self.discriminator(res_map)
        return logits

# ============================ CONFIG (paper values) =========================
SEED = 42
EPOCHS = 50               # paper: "a predetermined number of epochs, 50"
BATCH_SIZE = 16           # paper: batch size 16 (drop to 8 if CUDA OOM)
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]

# AdamW differential learning rates (paper Sec. 3.5.1)
FUSION_LR = 1e-3          # classifier + fusion gate
LCA_LR = 5e-4             # local cross-attention module
BACKBONE_LR = 1e-5        # pretrained ResNet-50 and ViT
DISCRIMINATOR_LR = 1e-3   # paper gives no LR for the discriminator
WEIGHT_DECAY = 1e-4       # paper mentions weight decay, not its value

# Progressive fine-tuning schedule (paper Sec. 3.5.4, values chosen here)
UNFREEZE_RESNET_AT = 15   # E1: deeper ResNet layers (layer3/layer4) unfreeze
RESNET_FULL_AT = 20       # rest of ResNet unfreezes (progressive step)
UNFREEZE_VIT_AT = 30      # E2: ViT unfreezes -> end-to-end training

# Stochastic domain discriminator (paper Sec. 3.4.4, Eq. 14)
USE_DOMAIN_LOSS = True
DOMAIN_LOSS_WEIGHT = 0.2

SMOKE_TEST = False        # True -> 1 epoch on 32 samples (quick sanity run)
SAVE_DIR = Path("rvit_checkpoint")
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]   # paper: ImageNet statistics
STD = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================== DATA ========================================
class BrainTumorDataset(Dataset):
    """Holds the [N,3,224,224] float tensors from Training.pt / Testing.pt.

    Paper Sec. 3.2.1 augmentation is applied on demand for the training
    split; validation/test only get the deterministic normalization.
    """

    def __init__(self, images, labels, train: bool):
        self.images = images
        self.labels = labels
        if train:
            self.transform = v2.Compose([
                v2.RandomResizedCrop(224, scale=(0.8, 1.0)),  # paper scale (0.8, 1.0)
                v2.RandomHorizontalFlip(),                    # 50% h-flip
                v2.RandomVerticalFlip(),                      # 50% v-flip
                v2.RandomRotation(15),                        # up to 15 deg
                v2.ColorJitter(brightness=0.3, contrast=0.3), # +-30%
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
    x_train, y_train = train_blob["images"], train_blob["labels"].long().view(-1)
    del train_blob
    test_blob = torch.load("Testing.pt", weights_only=True, map_location="cpu")
    x_test, y_test = test_blob["images"], test_blob["labels"].long().view(-1)
    del test_blob

    for name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        assert x.ndim == 4 and x.shape[1:] == (3, 224, 224), (name, x.shape)
        assert len(x) == len(y)
        assert torch.all((y >= 0) & (y < NUM_CLASSES)), name

    # 80/20 stratified validation split of the training set
    idx = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y_train.numpy()
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


# ==================== PROGRESSIVE UNFREEZING (Sec. 3.5.4) ===================
def apply_unfreeze_schedule(model, epoch: int) -> str:
    """Toggle requires_grad according to the paper's staged schedule.

    epoch <  E1 (15): all backbone layers frozen -- only LCA, fusion and
                      classifier train (paper: "early epochs").
    E1 <= epoch < 20 : deeper ResNet layers (layer3/layer4) unfrozen (paper:
                      "deeper layers ... progressively unfrozen after E1").
    20 <= epoch < E2 (30): whole ResNet trainable, ViT still frozen.
    epoch >= E2      : ViT unfrozen too -> end-to-end training (paper: "at a
                      later epoch E2").
    """
    resnet = model.resnet
    vit = model.vit

    def set_frozen(module, flag):
        for p in module.parameters():
            p.requires_grad_(flag)

    if epoch < UNFREEZE_RESNET_AT:
        for part in (resnet.stem, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4):
            set_frozen(part, False)
        set_frozen(vit, False)
        return "backbones frozen (LCA + fusion + classifier)"
    elif epoch < RESNET_FULL_AT:
        for part in (resnet.stem, resnet.layer1, resnet.layer2):
            set_frozen(part, False)
        for part in (resnet.layer3, resnet.layer4):
            set_frozen(part, True)
        set_frozen(vit, False)
        return "ResNet layer3/layer4 unfrozen"
    elif epoch < UNFREEZE_VIT_AT:
        set_frozen(resnet, True)
        set_frozen(vit, False)
        return "ResNet fully unfrozen"
    else:
        set_frozen(resnet, True)
        set_frozen(vit, True)
        return "end-to-end (ResNet + ViT)"


def _eval_frozen_parts(model):
    """Keep fully-frozen backbones in eval mode so BN stats stay fixed."""
    resnet = model.resnet
    for part in (resnet.stem, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4):
        if not any(p.requires_grad for p in part.parameters()):
            part.eval()
    if not any(p.requires_grad for p in model.vit.parameters()):
        model.vit.eval()


# ========================= TRAIN / EVAL LOOPS ===============================
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    _eval_frozen_parts(model)
    use_amp = DEVICE.type == "cuda"
    running_loss = correct = total = 0
    for images, labels in tqdm(loader, desc="Training", unit="batch", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            if USE_DOMAIN_LOSS:
                logits, domain_logits = model(images, return_domain=True)
                # paper: randomized binary domain labels created each cycle
                domain_targets = torch.randint(0, 2, domain_logits.shape, device=DEVICE).float()
                loss = criterion(logits, labels) + DOMAIN_LOSS_WEIGHT * nn.functional.binary_cross_entropy_with_logits(
                    domain_logits, domain_targets)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    """Train with the paper's optimizer/schedule and save the best checkpoint.

    Returns (history, best_state, best_val_acc).
    """
    optimizer = torch.optim.AdamW([
        {"params": model.resnet.parameters(), "lr": BACKBONE_LR},
        {"params": model.vit.parameters(), "lr": BACKBONE_LR},
        {"params": model.lca.parameters(), "lr": LCA_LR},
        {"params": model.fusion.parameters(), "lr": FUSION_LR},
        {"params": model.classifier.parameters(), "lr": FUSION_LR},
        *([{"params": model.discriminator.parameters(), "lr": DISCRIMINATOR_LR}]
          if USE_DOMAIN_LOSS else []),
    ], weight_decay=WEIGHT_DECAY)
    # paper Sec. 3.5.2: cosine annealing over the training epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    epochs = 1 if SMOKE_TEST else EPOCHS
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state, best_acc, best_loss = None, -1.0, float("inf")

    for epoch in tqdm(range(1, epochs + 1), desc="Fine-tuning", unit="epoch"):
        phase = apply_unfreeze_schedule(model, epoch)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch}/{epochs} [{phase}] | train {tr_loss:.4f}/{tr_acc*100:.2f}% "
              f"| val {va_loss:.4f}/{va_acc*100:.2f}%")

        # paper Sec. 3.5.6: save weights when validation accuracy improves
        if va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss):
            best_acc, best_loss = va_acc, va_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"epoch": epoch, "model_state_dict": best_state,
                 "val_acc": va_acc, "val_loss": va_loss, "train_loss": tr_loss},
                SAVE_DIR / "rvit_best.pth",
            )

    return history, best_state, best_acc


def plot_results(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    n = len(history["train_loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
    # progressive unfreezing stages (paper Sec. 3.5.4) as vertical markers
    stages = [
        (UNFREEZE_RESNET_AT, "E1: ResNet layer3/4\nunfrozen"),
        (RESNET_FULL_AT, "ResNet fully\nunfrozen"),
        (UNFREEZE_VIT_AT, "E2: ViT unfrozen\n(end-to-end)"),
    ]
    for pos, label in stages:
        if pos <= n:
            for ax in (ax1, ax2):
                ax.axvline(pos, linestyle="--", color="gray", linewidth=1)
            ax2.annotate(label, xy=(pos, 0.02), xycoords=("data", "axes fraction"),
                         fontsize=7, ha="center", color="gray")
    ax1.legend(); ax2.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "curves.png", dpi=150)
    plt.show()


def main():
    SAVE_DIR.mkdir(exist_ok=True)
    print("Device:", DEVICE)
    train_loader, val_loader, test_loader = build_loaders()

    model = RViTFusionNet(NUM_CLASSES).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Domain loss: {'on (lambda=0.2)' if USE_DOMAIN_LOSS else 'off'}")

    history, best_state, best_val = fit(model, train_loader, val_loader)
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
    plt.title("RViT-FusionNet Brain Tumor Classification")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "confusion_matrix.png", dpi=150)
    plt.show()

    plot_results(history)
    print(f"\nBest checkpoint saved to {SAVE_DIR / 'rvit_best.pth'}")
    print("Evaluate with: python RVIT_evaluate.py --split Testing.pt")


if __name__ == "__main__":
    main()
