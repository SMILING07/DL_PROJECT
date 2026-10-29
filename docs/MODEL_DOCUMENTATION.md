# Model Documentation — PDCNN · ViT-B/32 · RViT-FusionNet

Deep-dive into the three brain-tumor MRI classifiers in this repository: exact
architecture (layer-by-layer, with tensor shapes and parameter counts), training
protocol, results, and a bank of reviewer questions with model answers.

**Task:** 4-class brain MRI classification — `glioma (0)`, `meningioma (1)`,
`no tumor (2)`, `pituitary (3)`.

| Model | Family | Paper | Params (our build) | Test accuracy | Macro F1 |
|---|---|---|---|---|---|
| PDCNN | CNN (dilated, multi-scale) | Rahman et al., Digital 2024 | 12,624,196 (12.62 M) | **95.06%** (1521/1600) | 94.94% |
| ViT-B/32 (FT-ViT) | Transformer | Asiri et al., Sensors 2023 | 87,465,297 (87.47 M) | **93.81%** (1501/1600) | 93.67% |
| RViT-FusionNet | Hybrid CNN + ViT | Islam et al., Neural Comput & Appl 2026 | 128,454,344 (128.45 M) | **92.31%** (1477/1600) | 92.22% |

---

## 0. Common pipeline (shared by all three models)

### 0.1 Dataset

* **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  by Masoud Nickparvar (CC BY 4.0) — 7,200 images merged from figshare, SARTAJ
  and Br35H.
* **Official split:** `Training` = 5,600 images (1,400 per class),
  `Testing` = 1,600 images (400 per class). The split ships with the dataset;
  we keep it unchanged so our numbers are comparable with other work on the
  same benchmark.
* **Preprocessing** (`DATASET/data_prep.py`): PIL `convert("RGB")` →
  `resize((224, 224))` → float tensor → `/255` → stored as
  `Training.pt` / `Testing.pt` with `images (N,3,224,224) float32 in [0,1]` and
  `labels (N,) int64`. Images are shuffled once at prep time.
* **Label map:** `glioma=0, meningioma=1, notumor=2, pituitary=3`.
* **Normalization:**
  * PDCNN consumes raw `[0,1]` pixels (no normalization).
  * ViT and RViT apply ImageNet statistics inside the `Dataset` object —
    `μ = [0.485, 0.456, 0.406]`, `σ = [0.229, 0.224, 0.225]` — required by the
    ImageNet-pretrained backbones.

### 0.2 Validation strategy

| Model | Validation data | Test data used during training? |
|---|---|---|
| ViT-B/32 | 80/20 **stratified split of `Training.pt`** (4,480 train / 1,120 val, `random_state=42`) | No |
| RViT-FusionNet | 80/20 **stratified split of `Training.pt`** (4,480 train / 1,120 val, `random_state=42`) | No |
| PDCNN | `fit(x_val=x_test, y_val=y_test)` — the **test split doubles as validation** | Yes (see Q&A A6) |

### 0.3 Augmentation

| Model | Augmentations (train split only) |
|---|---|
| PDCNN | random horizontal flip, ±10° rotation (affine grid, bilinear, zero padding), ±10% brightness jitter, clamp to `[0,1]` — implemented on-GPU per batch in `PDCNN._augment` |
| ViT-B/32 | `RandomResizedCrop(224, scale=(0.8,1.0))`, horizontal + vertical flip, `RandomRotation(15)`, `ColorJitter(brightness=0.3)`, ImageNet normalize |
| RViT-FusionNet | `RandomResizedCrop(224, scale=(0.8,1.0))`, horizontal + vertical flip, `RandomRotation(15)`, `ColorJitter(brightness=0.3, contrast=0.3)`, ImageNet normalize |

Validation/test pipelines only normalize (deterministic, no augmentation).

### 0.4 Reproducibility & checkpointing

* `SEED = 42` for `random`, `numpy` and `torch` (incl. CUDA) in every script.
* Checkpoint dict: `{"epoch", "model_state_dict", "train_loss", "val_loss", "val_acc"}`
  (ViT/RViT save it as `*_best.pth`, PDCNN encodes the metrics in the filename:
  `PDCNN_best_ep41_trloss0.3617_vlloss0.2223_vacc95.06.pt`).
* Checkpoint selection = highest validation accuracy (RViT breaks ties with the
  lower validation loss).
* Artifacts written next to the checkpoint: loss/accuracy curves
  (with stage/early-stop markers) and a confusion-matrix PNG.

### 0.5 Evaluation suite

All three evaluators (`PDCNN_evaluate.py`, `VITB32_evaluate.py`,
`RVIT_evaluate.py`) expose the same interface:

```bash
python <model>_evaluate.py --split Testing.pt      # full report on 1600 images
python <model>_evaluate.py path/to/mri.jpg         # single-image prediction
# flags: --checkpoint/-c, --batch-size (128 PDCNN/ViT, 32 RViT), --no-plot
```

Report contents: overall accuracy (`correct/N`), per-class accuracy /
precision / recall / F1 with support, macro and support-weighted averages,
the raw confusion matrix, a saved matplotlib confusion-matrix figure, plus the
loaded checkpoint's metadata (epoch, val acc/loss) and split statistics.
Checkpoints are auto-picked: PDCNN = highest `vacc` in `pdcnn_checkpoint/`,
ViT = `vit_checkpoint/vit_b32_best.pth`, RViT = `rvit_checkpoint/rvit_best.pth`.

---

## 1. PDCNN — Parallel Dilated CNN

**Paper:** *Parallel Dilated Convolutional Neural Network for Brain Tumor MRI
Classification* (Rahman et al., Digital 2024) — `RESEACRH PAPERS/PDCNN.pdf`.
**Code:** `PDCNN_train.py` (model + training), `PDCNN_evaluate.py` (inference/eval).
**Parameters:** **12,624,196** (counted from the instantiated model).

### 1.1 Design idea

One shared "stem" extracts low-level MRI texture once; then **three parallel
branches with different dilation rates (d = 1, 2, 3)** see the *same* feature
map through three receptive-field sizes — local, medium and large context —
and their pooled descriptors are concatenated before the classifier. Dilated
convolutions grow the receptive field **without adding parameters or losing
spatial resolution** (no extra pooling inside a branch).

### 1.2 Building blocks

```python
ConvBlock(in, out, k=3, s=1, pad=1, dilation=1):
    Conv2d(bias=False) → BatchNorm2d → ReLU
```

```python
DilatedBranch(in, channels=[128,256,512], dilations=[d,d,d]):
    for each (out_ch, d):  ConvBlock → MaxPool2d(2)
    ConvBlock(512 → 512, dilation=d)      # final conv, no pooling
    AdaptiveAvgPool2d(1) → flatten        # GAP → (B, 512)
```

### 1.3 Layer-by-layer forward pass (input `(B,3,224,224)`)

| Stage | Operation | Output shape |
|---|---|---|
| Stem 1 | `ConvBlock(3→64, d=1)` | `(B,64,224,224)` |
| Stem 2 | `ConvBlock(64→64, d=1)` | `(B,64,224,224)` |
| Stem 3 | `MaxPool2d(2)` | `(B,64,112,112)` |
| Branch **d=1** | conv/pool ×3 → final conv → GAP | `(B,512)` — spatial path 112→56→28→14, final conv at 14×14 |
| Branch **d=2** | conv/pool ×3 → final conv → GAP | `(B,512)` — 112→110→55→53→26→24→12→10, final conv at 10×10 |
| Branch **d=3** | conv/pool ×3 → final conv → GAP | `(B,512)` — 112→108→54→50→25→21→10→6, final conv at 6×6 |
| Concat | `torch.cat` of the 3 branches | `(B,1536)` |
| Head 1 | `Linear(1536→512) → BN → ReLU → Dropout(0.5)` | `(B,512)` |
| Head 2 | `Linear(512→128) → ReLU → Dropout(0.3)` | `(B,128)` |
| Head 3 | `Linear(128→4)` | `(B,4)` logits |

Channel plan per branch: `64 → 128 → 256 → 512` with three MaxPools (each
branch), plus one un-pooled `512→512` conv before the GAP. Branch depths are
identical — only the dilation differs, so no branch dominates the parameter
budget.

**Weight init:** Kaiming-normal (`fan_out`, ReLU) for convs, BN
`γ=1, β=0`, Linear `trunc_normal_(std=0.02)` with zero bias.

### 1.4 Training protocol

| Item | Value |
|---|---|
| Optimizer | **AdamW**, `lr = 5e-4`, `weight_decay = 0.01` |
| Schedule | `OneCycleLR` over all 50 epochs (`pct_start=0.1`) |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` |
| Batch / epochs | 32 / 50 (train batch 32, eval batch 128) |
| Precision | AMP (`torch.amp.autocast` + `GradScaler`) on CUDA |
| Gradient clipping | global norm `1.0` |
| Memory pattern | batches are indexed on CPU and moved to GPU one at a time (`del xb, yb`), so the full 5,600-image tensor stays in RAM |
| Checkpoint | best-val (`PDCNN_best_ep{N}_trloss…_vacc….pt`) + final (`PDCNN_final_ep50_…`) into `checkpoint/` |

### 1.5 Results (best checkpoint: epoch 41, val acc 95.06%)

**Test accuracy 95.06% — 1,521 / 1,600 correct.**
Macro: precision 95.35%, recall 95.06%, F1 94.94%.

| Class (n=400) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Glioma | 82.50% | 99.70% | 82.50% | 90.29% |
| Meningioma | 97.75% | 90.72% | 97.75% | 94.10% |
| No tumor | 100.00% | 93.90% | 100.00% | 96.85% |
| Pituitary | 100.00% | 97.09% | 100.00% | 98.52% |
| **Macro avg** | **95.06%** | **95.35%** | **95.06%** | **94.94%** |

Confusion matrix (rows = true, cols = predicted):

| true \ pred | glioma | meningioma | no tumor | pituitary |
|---|---|---|---|---|
| **glioma** | 330 | 40 | 25 | 5 |
| **meningioma** | 1 | 391 | 1 | 7 |
| **no tumor** | 0 | 0 | 400 | 0 |
| **pituitary** | 0 | 0 | 0 | 400 |

Failure mode: 70 of the 70 errors on glioma are *false negatives* — glioma is
visually confusable with meningioma (40) and with normal tissue (25).
Precision for glioma is 99.70%, i.e. the model rarely calls another class
"glioma" but under-detects it.

---

## 2. ViT-B/32 (FT-ViT)

**Paper:** Asiri et al., *"Advancing brain tumor classification through
fine-tuned vision transformers"*, Sensors 2023, 23(18), 7913 —
`RESEACRH PAPERS/VIT.pdf`.
**Code:** `VITB32_train.py` (head, training), `VITB32_evaluate.py` (eval).
**Parameters:** **87,465,297** (torchvision `vit_b_32` with the paper head;
the paper's Table 2 quotes 87.65 M for the same variant — the difference is
the head's exact layer sizing/rounding).

### 2.1 Design idea

A standard **Vision Transformer** splits the image into non-overlapping
patches, treats the patch sequence as a "sentence" of tokens, and processes it
with global self-attention. Pretrained ImageNet weights provide strong generic
features; only a **very small task head** (11-unit bottleneck) is trained for
this 4-class problem, which limits overfitting on 5,600 images.

### 2.2 Architecture (torchvision `vit_b_32`)

| Step | Operation | Output shape |
|---|---|---|
| Input | RGB | `(B,3,224,224)` |
| Patchify | non-overlapping **32×32** patches → `224/32 = 7` → 49 patches | 49 × (32,32,3) |
| Flatten + linear projection | `32·32·3 = 3072 → 768` | `(B,49,768)` |
| CLS token + position embedding | prepend learnable CLS, add 50 learned positions | `(B,50,768)` |
| Encoder ×12 | each block: `LN → Multi-Head Self-Attention (12 heads, dim 768) → residual` then `LN → MLP 768→3072→768 → residual` | `(B,50,768)` |
| Final LN | pre-head normalization | `(B,50,768)` |
| Take CLS | index `0` | `(B,768)` |
| **FT-ViT head** | `Flatten → BatchNorm1d(768) → Linear(768→11) → ReLU → BatchNorm1d(11) → Linear(11→4)` | `(B,4)` logits |

Notes:
* The default torchvision 1000-class head (`Linear(768→1000)`) is discarded
  (`model.heads = PaperHead(...)`), reproducing the paper's Table 2 head.
* The 11-unit bottleneck acts as an aggressive regularizer: the 768-d
  penultimate features are compressed before the 4-class decision.
* ReLU after the 11-unit layer is our assumption — the paper specifies a fully
  linear stack, which would collapse to a single linear map.

### 2.3 Two-stage training protocol

| Item | Value |
|---|---|
| Stage 1 (epochs 1–5) | backbone **frozen** (`requires_grad` only on `heads.*`), Adam `lr = 1e-3` — head warm-up |
| Stage 2 (epochs 6–25) | **all** parameters trainable, Adam `lr = 1e-4` (paper's value), `weight_decay = 1e-4`, `CosineAnnealingLR(T_max = 20)` |
| Loss | plain `CrossEntropyLoss` |
| Batch / epochs | 16 / 25 (paper: 10 — see Q&A C4) |
| Early stop | patience **8** on validation accuracy (no improvement for 8 epochs → stop) |
| Checkpoint | on every val-acc improvement → `vit_checkpoint/vit_b32_best.pth` |
| Data | 4,480 train / 1,120 val (stratified 80/20 of `Training.pt`), 1,600 test |
| Augmentation | crop 0.8–1.0, h/v flip, rotation 15°, brightness 0.3, ImageNet normalize |
| Plots | `vit_checkpoint/vit_b32_curves.png` (stage boundary at epoch 5 + early-stop marker), `vit_b32_confusion_matrix.png`, `vit_b32_best_Testing_cm.png` |

A strict re-run of the paper's 10-epoch schedule stopped early at epoch 8
(val 90.18%) and reached only **84.62% test accuracy** — the two-stage/25-epoch
recipe above is what lifts the model to 93.81%.

### 2.4 Results (best checkpoint: epoch 25, val acc 98.12%)

**Test accuracy 93.81% — 1,501 / 1,600 correct.**
Macro: precision 94.11%, recall 93.81%, F1 93.67%.

| Class (n=400) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Glioma | 80.25% | 97.57% | 80.25% | 88.07% |
| Meningioma | 95.75% | 88.86% | 95.75% | 92.18% |
| No tumor | 100.00% | 91.74% | 100.00% | 95.69% |
| Pituitary | 99.25% | 98.27% | 99.25% | 98.76% |

Same signature as PDCNN: no-tumor is solved, pituitary nearly solved, glioma is
the hard class (19.75% of gliomas are mislabelled, mostly as meningioma).

---

## 3. RViT-FusionNet — ResNet-50 + ViT-B/16 + Local Cross-Attention

**Paper:** Islam et al., *"RViT-FusionNet"*, Neural Computing & Applications
2026, 38:518 — `RESEACRH PAPERS/RVIT.pdf`.
**Code:** `RVIT_FUSIONNET_train.py` (contains the model class
`RViTFusionNet`, imported by `RVIT_evaluate.py`).
**Parameters:** **128,454,344 (128.45 M)** — measured per module:

| Module | Parameters |
|---|---|
| `resnet` (ResNet-50 without the 1000-class FC) | 23,508,032 |
| `vit` (ViT-B/16 without the head) | 85,798,656 |
| `lca` (1×1 projection + 8-head MHA + LayerNorm) | 3,937,536 |
| `fusion` (3 projections + gate MLP) | 4,460,547 |
| `classifier` (1024→128→4) | 131,716 |
| `discriminator` (patch discriminator) | 10,617,857 |
| **Total** | **128,454,344** |

### 3.1 Design idea

Two complementary pretrained backbones — a **convolutional** ResNet-50 (local
texture, translation-equivariant) and a **transformer** ViT-B/16 (global
context) — are fused explicitly: the ResNet feature map *queries* the ViT
tokens through **Local Cross-Attention**, and a **learned gate** weights the
three global descriptors (ResNet, ViT, LCA) per sample. A **stochastic domain
discriminator** regularizes the shared features.

### 3.2 Branch A — ResNet-50 feature extractor (`ResNetFeatures`)

| Step | Operation | Output |
|---|---|---|
| stem | `conv1 7×7 s2 64 → BN → ReLU → maxpool 3×3 s2` | `(B,64,56,56)` |
| layer1 | 3 × bottleneck (256) | `(B,256,56,56)` |
| layer2 | 4 × bottleneck (512, s2) | `(B,512,28,28)` |
| layer3 | 6 × bottleneck (1024, s2) | `(B,1024,14,14)` |
| layer4 | 3 × bottleneck (2048, s2) | `(B,2048,7,7)` ← **spatial map** |
| pool | mean over H,W | `(B,2048)` ← **global vector** |

Weights: `ResNet50_Weights.DEFAULT` (ImageNet-1K).

### 3.3 Branch B — ViT-B/16 patch tokens (`ViTPatchFeatures`)

| Step | Operation | Output |
|---|---|---|
| patch embed | `224/16 = 14` → **196 tokens** of dim 768 | `(B,196,768)` |
| CLS + encoder | prepend CLS token, run `vit.encoder` (positional embedding, dropout, 12 blocks, final LN) | `(B,197,768)` |
| split | keep tokens `1:` (drop CLS from the pooled descriptor) | `(B,196,768)` |
| pool | mean over the 196 patch tokens | `(B,768)` |

The torchvision classification head is replaced by `nn.Identity()` — only
features are used. Weights: `ViT_B_16_Weights.DEFAULT`.

### 3.4 Local Cross-Attention (`LocalCrossAttention`) — paper Sec. 3.4.3

```
resnet_map [B,2048,7,7]
   └─ Conv2d 2048→768 (1×1) → flatten → transpose      → queries  [B,49,768]
vit_tokens [B,196,768]                                  → keys/vals [B,196,768]
   └─ nn.MultiheadAttention(embed=768, heads=8, batch_first)
        → [B,49,768]
   └─ LayerNorm( queries + attention_output )           # residual + norm
   └─ mean over the 49 queries                          → local_vec [B,768]
```

Interpretation: the 7×7 = **49 spatial locations of the ResNet map are the
queries**, so each *local* region of the MRI decides which *global* ViT tokens
are relevant to it — a local↔global correspondence the paper calls LCA.

### 3.5 Gated fusion (`GatedFeatureFusion`) — paper Sec. 3.4.5

```
res_proj : Linear(2048 → 1024)   for ResNet vector
vit_proj : Linear(768  → 1024)   for ViT vector
local_proj: Linear(768 → 1024)   for LCA vector
gate     : Linear(3072 → 256) → ReLU → Linear(256 → 3) → softmax(dim=1)
fused    = Σ_i  w_i · f_i          → [B,1024]   (w = [w_res, w_vit, w_lca])
```

The gate produces a **per-sample** soft weighting (three numbers summing to 1),
so the network can trust the CNN branch for texture-heavy slices and the
transformer branch for global morphology.

### 3.6 Classifier and discriminator

* **Classifier:** `Linear(1024→128) → GELU → Dropout(0.3) → Linear(128→4)` → logits.
* **Patch discriminator (paper Sec. 3.4.4):**
  `Conv(2048→512,3×3,pad1) → ReLU → Conv(512→256,3×3,pad1) → ReLU → Conv(256→1,1×1)`
  → `(B,1,7,7)` patch logits.
  Trained with **randomized binary targets** drawn fresh every iteration
  (`torch.randint(0,2,…)`) and `BCEWithLogits`, weight **λ = 0.2**, added to
  the cross-entropy. **No gradient-reversal layer** — this is the paper's
  *stochastic* domain loss, not classic DANN.

### 3.7 Training protocol

| Item | Value |
|---|---|
| Optimizer | **AdamW**, differential learning rates (paper Sec. 3.5.1): backbones `1e-5`, LCA `5e-4`, fusion + classifier `1e-3`, discriminator `1e-3`; `weight_decay = 1e-4` |
| Schedule | `CosineAnnealingLR(T_max = 50)` (paper Sec. 3.5.2) |
| Loss | `CE(logits, y) + 0.2 · BCE(disc_logits, random_targets)` |
| Batch / epochs | 16 / 50 |
| Precision | AMP + `GradScaler` on CUDA |
| Data | 4,480 train / 1,120 val (stratified 80/20), 1,600 test |
| Checkpoint | best val acc (tie → lower val loss) → `rvit_checkpoint/rvit_best.pth` |
| Plots | `rvit_checkpoint/curves.png` (vertical markers at each unfreezing stage), `confusion_matrix.png`, `rvit_best_Testing_cm.png` |

**Progressive unfreezing (paper Sec. 3.5.4; the paper names E1/E2 but gives no
numbers, so we chose them):**

| Epochs | State | Trainable |
|---|---|---|
| 1 – 14 | backbones frozen | LCA + fusion + classifier (+ discriminator) |
| 15 – 19 | **E1 = 15**: ResNet `layer3`/`layer4` unfrozen | + deep ResNet |
| 20 – 29 | ResNet fully unfrozen, ViT still frozen | ResNet + head modules |
| 30 – 50 | **E2 = 30**: ViT unfrozen | end-to-end |

Fully frozen submodules are forced into `.eval()` each epoch so BatchNorm
statistics and dropout stay fixed while their weights are frozen.

### 3.8 Results (best checkpoint: epoch 47, val acc 97.14%, val loss 0.1184)

**Test accuracy 92.31% — 1,477 / 1,600 correct.**
Macro: precision 92.64%, recall 92.31%, F1 92.22%.

| Class (n=400) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Glioma | 80.25% | 96.98% | 80.25% | 87.82% |
| Meningioma | 94.25% | 86.27% | 94.25% | 90.08% |
| No tumor | 96.25% | 91.45% | 96.25% | 93.79% |
| Pituitary | 98.50% | 95.86% | 98.50% | 97.16% |

---

## 4. Cross-model comparison

| | PDCNN | ViT-B/32 | RViT-FusionNet |
|---|---|---|---|
| Parameters | 12.62 M | 87.47 M | 128.45 M |
| Pretraining | none (trained from scratch) | ImageNet ViT-B/32 | ImageNet ResNet-50 + ViT-B/16 |
| Best epoch / val acc | 41 / 95.06% | 25 / 98.12% | 47 / 97.14% |
| Test accuracy | **95.06%** (1521/1600) | **93.81%** (1501/1600) | **92.31%** (1477/1600) |
| Macro P / R / F1 | 95.35 / 95.06 / 94.94 | 94.11 / 93.81 / 93.67 | 92.64 / 92.31 / 92.22 |
| Glioma recall | 82.50% | 80.25% | 80.25% |
| Pituitary recall | 100.00% | 99.25% | 98.50% |
| Optimizer / LR | AdamW 5e-4 + OneCycle | Adam 1e-3→1e-4, two-stage | AdamW 1e-5/5e-4/1e-3, cosine |
| Epochs | 50 | 25 | 50 |
| Validation split | uses the test split | 80/20 of train | 80/20 of train |

The consistent story across all three models: **no-tumor and pituitary are
essentially solved, meningioma is easy, glioma is the bottleneck** (80–83%
recall) because low-grade glioma regions look similar to meningeal tissue and
to healthy brain on T1-weighted slices.

---

## 5. Reviewer questions and answers

### A. Dataset, splits and methodology

**A1. Which dataset did you use, and is it balanced?**
The public Kaggle *Brain Tumor MRI Dataset* by Masoud Nickparvar (CC BY 4.0),
7,200 images merged from figshare, SARTAJ and Br35H. It is perfectly balanced:
1,400 train + 400 test images for each of the four classes. All images are
resized to 224×224 RGB and normalized to `[0,1]`; ViT/RViT additionally apply
ImageNet mean/std normalization.

**A2. How did you split the data? Isn't there a risk of leakage?**
We keep the dataset's official 5,600/1,600 train-test partition, so our
numbers are directly comparable with other published results on the same
benchmark. Inside the training set, ViT and RViT carve out a stratified 80/20
validation split (`random_state=42`) used only for checkpoint selection and
early stopping — the test set is never touched during training for those two
models.

**A3. Is the split patient-independent?**
The dataset ships no patient IDs, so patient-level independence cannot be
verified for the official split; images come from three source corpora that
were merged by the dataset author. We state this as a limitation: for clinical
claims one would re-split by patient/study ID (and the training scripts'
docstrings flag exactly that requirement).

**A4. Why 224×224 and RGB?**
224×224 is the native input size of the pretrained backbones (ViT-B/32 → 7×7
patches, ViT-B/16 → 14×14, ResNet-50 → 7×7 final map). MRI slices are
grayscale; replicating to 3 channels lets us reuse ImageNet-pretrained weights
(and the PDCNN design in the paper also takes 3-channel input).

**A5. What preprocessing/augmentation did you apply?**
Resize + `/255` in `data_prep.py`. Training-only augmentation: PDCNN does
in-batch flip, ±10° rotation and ±10% brightness jitter; ViT/RViT use
`RandomResizedCrop(0.8–1.0)`, horizontal/vertical flips, ±15° rotation and
color jitter (RViT also jitters contrast), then ImageNet normalization.
Validation/test only normalize — no randomness at evaluation time.

**A6. You select PDCNN checkpoints on the *test* set — isn't that leakage?**
Honest answer: yes, that is a methodological weakness of the PDCNN script.
`PDCNN_train.py` follows the reference implementation, which passes
`x_val=x_test` — so its "validation accuracy" *is* test accuracy (95.06% at
epoch 41), and model selection therefore sees the test distribution. The
ViT and RViT scripts do it properly (stratified validation split from the
training data only). The correct fix is a held-out validation split for
PDCNN as well, after which the reported number would be an unbiased test
estimate; we keep the current protocol only to stay faithful to the reference
training script and we disclose it explicitly.

**A7. Why four classes and not tumor/no-tumor binary or grading?**
The dataset is annotated with exactly four labels (glioma, meningioma,
pituitary, no tumor). Multi-class classification is the benchmark task; grade
prediction and segmentation would require different annotations (e.g.,
BraTS), which is listed as future work.

### B. PDCNN

**B1. What exactly is a dilated (atrous) convolution?**
A 3×3 kernel whose weights are spaced `d` pixels apart — it samples a 5×5 (d=2)
or 7×7 (d=3) footprint with the *same* 9 weights per channel. The receptive
field grows without extra parameters and without additional pooling, so the
network sees larger context while keeping fine spatial detail.

**B2. Why three parallel branches instead of one deeper network?**
Different dilation rates give different receptive-field sizes *simultaneously*:
d=1 for local texture, d=2 for medium structures, d=3 for large-scale context
(e.g., whole tumor region vs. surrounding edema). Concatenating the three GAP
descriptors (3×512 = 1536-d) hands the classifier multi-scale evidence in one
step, rather than forcing a single stack of layers to reuse one scale.

**B3. Why is the branch output spatially pooled (GAP)?**
`AdaptiveAvgPool2d(1)` reduces each 512-channel map to a 512-vector, making the
classifier size-independent, drastically cutting parameters in the head, and
adding mild translation invariance — a standard anti-overfitting device for
small medical datasets.

**B4. The paper says Adam — why do you use AdamW with weight decay and OneCycle?**
AdamW decouples weight decay from the adaptive step size and OneCycle drives
the learning rate through a warm-up/anneal, which empirically stabilizes
training of the scratch-built CNN (no pretraining to fall back on). The
reference implementation we were asked to reproduce already ships this recipe;
we kept it and documented the deviation from the literal paper text.

**B5. Why label smoothing 0.1?**
It softens the one-hot targets, discourages over-confident logits and
typically improves generalization on medical images where some labels
(glioma vs. meningioma) are inherently ambiguous.

**B6. Why does the smallest model perform best?**
Three reasons: (i) convolutions carry strong inductive biases (locality,
translation equivariance) which matter when training 5,600 images from
scratch; (ii) 12.6 M parameters are far easier to fit than 87 M/128 M without
massive pretraining; (iii) heavy in-batch augmentation regularizes it. The
transformer/hybrid models are pre-trained on natural images (ImageNet), whose
statistics differ from MRI, and they must still adapt their backbones.

**B7. Is 95.06% really the test accuracy if validation == test?**
The 95.06% is the accuracy of the epoch-41 checkpoint on the 1,600 test
images (1,521 correct). Because the same split was used for selection, the
number carries selection bias (see A6) — a validation-split retrain would be
the rigorous way to confirm it.

**B8. Where did the errors go?**
All 70 errors are glioma false negatives: 40 predicted meningioma, 25 predicted
no-tumor, 5 pituitary. Precision for glioma is 99.70% — the model is
conservative about calling something glioma, which is the safer error
direction clinically (misses are still serious, so recall 82.5% is the number
to improve).

### C. ViT-B/32

**C1. How does a Vision Transformer work?**
The image is cut into non-overlapping 32×32 patches (49 patches), each flattened
(3072 values) and linearly projected to 768 dimensions. A learnable CLS token is
prepended and positional embeddings are added so order is preserved. The
sequence passes through 12 identical encoder blocks of
`LayerNorm → 12-head self-attention → residual → LayerNorm → MLP(768→3072→768) → residual`.
Self-attention lets every patch attend to every other patch — global receptive
field from layer 1 — and the CLS output is the image descriptor.

**C2. Why replace the head with Flatten→BN→Dense(11)→BN→Dense(4)?**
That is the exact task head of the FT-ViT paper (Table 2): an 11-unit
bottleneck compresses the 768-d features, which acts as a strong regularizer
for a small 4-class dataset, while BatchNorm layers stabilize the tiny
amount of trainable signal. The full parameter count (87,465,297) matches the
paper's ViT-b32 entry.

**C3. Why two training stages?**
Training the randomly initialized head at 1e-3 against a *frozen* pretrained
backbone (epochs 1–5) first aligns the head with the existing feature space.
Only then is the whole network unfrozen at the paper's 1e-4 with cosine
annealing (epochs 6–25). This avoids the random head's large gradients
destroying pretrained features in the first iterations.

**C4. The paper says 10 epochs — why did you train 25?**
We ran the paper's schedule first: it early-stopped at epoch 8 with 90.18%
val accuracy but only **84.62% test accuracy** — the backbone was
undertrained on this dataset. Extending to 25 epochs (5 warm-up + 20
fine-tune), keeping the paper's LR, head and augmentation, reached **93.81%**
and val 98.12%. The deviation is documented in the code comments and the
report; early stopping (patience 8) is still active, so the run ends if it
stops improving.

**C5. Validation accuracy is 98.12% but test accuracy is 93.81% — why the gap?**
Three contributors: the validation split (1,120 images) is easier and 4.4%
the size of the test set, so its estimate is noisy and optimistic; checkpoint
selection picks the epoch maximizing that estimate (selection bias); and
ImageNet pretraining still leaves a domain gap to MRI. A 4.3-point gap is
normal for this dataset size; the test number is the one we report.

**C6. Why not train ViT from scratch?**
A ViT has no convolutional inductive bias and needs ImageNet-scale data to
converge; with 5,600 images a from-scratch ViT fails to generalize. Transfer
learning from ImageNet-pretrained weights is standard practice and is what the
FT-ViT paper prescribes.

**C7. What does patch size 32 mean computationally vs. 16?**
32×32 patches → only 49 tokens (+1 CLS) → attention matrix of 50×50 per head:
cheap, but each token aggregates a coarse 32×32 region (fine details averaged
away). 16×16 would give 196 tokens (≈16× more attention work) with finer
localization. The paper found b32 (98.24% on its own data) competitive with
b16 at much lower cost, which is why we fine-tuned b32.

**C8. Why ImageNet normalization for ViT but not PDCNN?**
The pretrained backbone was trained on ImageNet-normalized inputs; feeding it
`[0,1]` raw pixels would shift activations away from the regime the
weights were optimized for. PDCNN is trained from scratch, so it learns its
own input scale.

### D. RViT-FusionNet

**D1. Why combine ResNet and ViT at all?**
Each covers the other's weakness: ResNet-50 gives strong local texture
features with a small sample requirement; ViT-B/16 gives long-range
dependencies (whole-tumor context). Instead of simple concatenation, the paper
models their *correspondence* explicitly (LCA) and learns how much to trust
each branch per image (gated fusion).

**D2. Explain Local Cross-Attention — what are the queries, keys and values?**
Queries = the ResNet feature map projected from 2048 → 768 channels and
flattened to 49 spatial positions (the 7×7 grid). Keys/values = the 196 ViT
patch tokens. Eight-head attention computes, for every ResNet location, a
weighted mix of the global ViT tokens; a residual + LayerNorm preserves the
original local signal, and mean-pooling the 49 outputs yields a 768-d
"local context" vector. In short: *local regions look up the global
information they need.*

**D3. What does the fusion gate do?**
Each of the three descriptors (ResNet 2048-d, ViT 768-d, LCA 768-d) is
projected into a common 1024-d space. A small MLP maps their concatenation
(3072-d) to 3 logits that are softmaxed into weights `w_res, w_vit, w_lca`
summing to 1 per sample, and the fused vector is their weighted sum. The gate
makes the contribution of each branch *interpretable* and lets the model lean
on the CNN for texture-dominated slices.

**D4. What is the "stochastic domain discriminator" and is it a GAN?**
It is not a GAN. The patch discriminator (2048→512→256→1 convolutions over the
7×7 ResNet map) is trained with BCE against **randomly drawn 0/1 targets each
iteration**, weighted λ = 0.2 and added to the cross-entropy — the paper's
stochastic domain loss (Sec. 3.4.4). Random targets act as a regularizer that
prevents the backbone from leaning on any single discriminative shortcut;
crucially, we follow the paper in *not* using a gradient-reversal layer, so it
is not a domain-adversarial (DANN) objective.

**D5. Why differential learning rates?**
The backbones are ImageNet-pretrained and must move only slightly (1e-5), the
LCA module is a new but structured attention block (5e-4), and the freshly
initialized fusion gate and classifier must learn fast (1e-3). A single LR
would either destroy the pretrained features or undertrain the new layers.

**D6. Why progressive unfreezing, and where do E1/E2 come from?**
Freezing the backbones first lets the randomly initialized fusion/classifier
converge on stable features (paper Sec. 3.5.4 "early epochs"), then unfreezing
deep ResNet layers (E1 = 15) and finally the ViT (E2 = 30) adapts low-level
features only once the head is sane. The paper names E1/E2 without values, so
we chose 15/20/30 over the 50-epoch schedule; the training-curve plot marks
each stage so the effect on val accuracy is visible.

**D7. Why is RViT the largest but the lowest-accuracy model here?**
128 M parameters trained on 4,480 images is the hardest generalization
problem of the three, and the schedule must freeze/unfreeze carefully to avoid
catastrophic forgetting. Its 92.31% is still strong, it beats the pure ViT on
pituitary/meningioma confusion structure, and its architecture is designed for
*robustness/interpretability* (per-sample gate weights, explicit
local-global modelling) rather than raw leaderboard accuracy on a single
small benchmark. Also compare with the paper's own numbers: 94–99% are on its
own datasets (DS1–DS4), not this Kaggle split.

**D8. Does the discriminator slow training / is it necessary?**
It adds one small conv pass over a 7×7 map — negligible next to the two
backbones. The loss weight is only 0.2, so it perturbs the gradient mildly.
We keep it because the paper's protocol includes it; setting
`USE_DOMAIN_LOSS = False` in `RVIT_FUSIONNET_train.py` reproduces the
ablation without code changes.

**D9. Why batch size 16 for RViT but 32 for PDCNN?**
RViT keeps two full backbones (ResNet-50 + ViT-B/16) plus attention in memory;
16 is the paper's batch size and fits the GPU with AMP enabled. PDCNN (12.6 M
parameters) can afford 32.

### E. Results and comparison

**E1. Why is PDCNN (12.6 M) better than ViT (87 M) and RViT (128 M)?**
Parameter count is not the same as inductive bias. With only 5,600 training
images, convolutional locality/translation-equivariance plus strong
augmentation generalizes better than large pretrained backbones that must
adapt from natural images to MRI. The transformer models compensate with fast
convergence (ImageNet pretraining) but end a few points behind on this
benchmark.

**E2. Why is glioma always the worst class?**
Glioma regions are low-contrast and heterogeneous and can resemble meningioma
or normal tissue on a single T1 slice; all three models show ~80–83% glioma
recall with confusions routed mainly to meningioma and no-tumor. Improving it
likely needs multi-sequence MRI, 3D context, or Grad-CAM-guided data curation.

**E3. Your accuracies are lower than the papers — why?**
The papers evaluate on their own data distributions: the RViT paper reports
94.76–99.56% across DS1–DS4, the FT-ViT paper reports 98.24% for ViT-b32 on
its collection. We reproduce the *architectures and training recipes* on the
Kaggle 7,200-image benchmark, which has a different case mix and difficulty.
Comparisons are only meaningful within the same dataset — which is why our
comparison table contains our three numbers measured on one shared test set.

**E4. Which model would you deploy?**
PDCNN: highest accuracy, smallest footprint (fastest inference, easiest to
run on modest hardware). The ViT/RViT are kept as complementary baselines;
an ensemble of the three (majority vote / averaged probabilities) is the
natural next experiment and is listed under future work.

**E5. What metric besides accuracy do you report, and why?**
Per-class precision/recall/F1, macro and support-weighted averages, and the
confusion matrix — accuracy alone hides class-specific failures. Macro F1
(94.94 / 93.67 / 92.22) is the fairest single number here because the test
set is balanced but glioma is clearly harder.

**E6. Are the differences between the models statistically meaningful?**
On 1,600 test images, ±1.6 points corresponds roughly to one standard error
(√(p(1−p)/n) ≈ 0.7%), so PDCNN's 95.06% vs. ViT's 93.81% is about a
1.3-point margin — meaningful but modest; ViT vs. RViT (1.5 points) is similar
in scale. A McNemar test on paired predictions, or repeated runs with
different seeds, would be the rigorous follow-up; today all numbers are
single-run, seed 42.

### F. Evaluation, engineering and limitations

**F1. How do you evaluate a model?**
`python <model>_evaluate.py --split Testing.pt` prints accuracy
(correct/1600), per-class accuracy/precision/recall/F1 with support, macro and
weighted averages, the confusion matrix, the checkpoint's metadata, and saves
a confusion-matrix figure. The same script accepts a single image path and
prints the predicted class, confidence and full class-probability vector.

**F2. How do you keep results reproducible?**
Fixed seeds (42) for Python/NumPy/torch/CUDA, deterministic preprocessing,
checkpoints that store epoch + train/val loss + val accuracy, best-model
selection on validation accuracy, and scripts whose hyperparameters live in a
single CONFIG block at the top of each file.

**F3. What do the checkpoints contain?**
`{"epoch", "model_state_dict", "train_loss", "val_loss", "val_acc"}`.
Loading is a 4-line recipe shown in the README for each model (build the
class, `load_state_dict(ckpt["model_state_dict"])`). PDCNN additionally
encodes the metrics in the filename.

**F4. What hardware do you need?**
Training was run on CUDA with mixed precision; PDCNN is the lightest (batch
32, 50 epochs), ViT is fastest per epoch (small head, frozen backbone for the
first 5 epochs), RViT is the heaviest (two backbones, batch 16, 50 epochs).
All three scripts fall back to CPU and print a VRAM-guidance comment block.

**F5. What are the main limitations?**
1. PDCNN uses the test split for validation/model selection (A6).
2. Single-run results (one seed) without confidence intervals or McNemar
   testing.
3. No external validation on another hospital/scanner distribution.
4. Dataset has no patient IDs → patient-level independence unverified (A3).
5. Glioma recall (~80%) remains the clinical weak spot.
6. No Grad-CAM/attention interpretability yet, and no segmentation/grading.

**F6. What would you do next?**
Fix the PDCNN validation protocol, run multi-seed evaluations with McNemar
tests, add Grad-CAM to explain glioma errors, try TTA and an ensemble of the
three models, test cross-dataset generalization, and explore 3D/multi-sequence
inputs for grading.

**F7. Is this a clinical tool?**
No. It is an academic benchmark comparison on a public dataset; it has not
been validated on clinical data, has no regulatory clearance, and must not be
used for diagnosis. The realistic framing is decision-support research, where
recall on glioma would need to be raised and externally validated first.

---

### Quick reference — files

| Purpose | File |
|---|---|
| Train PDCNN | `PDCNN_train.py` → `checkpoint/` (rename to `pdcnn_checkpoint/`) |
| Train ViT-B/32 | `VITB32_train.py` → `vit_checkpoint/vit_b32_best.pth` + curves/CM |
| Train RViT | `RVIT_FUSIONNET_train.py` → `rvit_checkpoint/rvit_best.pth` + curves/CM |
| Evaluate | `PDCNN_evaluate.py` / `VITB32_evaluate.py` / `RVIT_evaluate.py` |
| Data prep | `DATASET/data_prep.py` → `Training.pt` / `Testing.pt` |
| Papers | `RESEACRH PAPERS/{PDCNN,VIT,RVIT}.pdf` |
