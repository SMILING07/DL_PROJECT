# Brain Tumor MRI Classification — PDCNN · ViT-B/32 · RViT-FusionNet

Three models classifying brain MRI scans into 4 classes: **glioma**, **meningioma**, **no tumor**, **pituitary**.

| Model | Paper | Test accuracy | Params |
|---|---|---|---|
| PDCNN | Parallel Dilated CNN | **95.06%** | 12.6M |
| ViT-B/32 | Asiri et al., Sensors 2023 (FT-ViT) | **93.81%** | 87.5M |
| RViT-FusionNet | Islam et al., Neural Comput & Applic 2026 | **92.31%** | 128.5M |

---

## File structure

```
DL_MODELS/                          # run everything from this directory
├── PDCNN_train.py                  # train PDCNN            -> checkpoint/
├── PDCNN_evaluate.py               # evaluate PDCNN         (reads pdcnn_checkpoint/)
├── VITB32_train.py                 # train ViT-B/32         -> vit_checkpoint/vit_b32_best.pth
├── VITB32_evaluate.py              # evaluate ViT-B/32
├── RVIT_FUSIONNET_train.py         # train RViT-FusionNet   -> rvit_checkpoint/rvit_best.pth
├── RVIT_evaluate.py                # evaluate RViT-FusionNet
├── README.md
├── IEEE_Project_Report.docx        # project report (IEEE two-column)
├── PRESENTATION/                   # slides
├── RESEACRH PAPERS/                # source papers (PDCNN, VIT, RVIT pdf)
├── DATASET/
│   └── data_prep.py                # builds Training.pt / Testing.pt from raw images
└── (after data prep + training, created locally — not in git:)
    ├── DATASET/Training/...        # raw MRI images (Kaggle dataset)
    ├── DATASET/Testing/...
    ├── Training.pt / Testing.pt    # preprocessed tensors
    ├── pdcnn_checkpoint/           # best PDCNN checkpoint + figures
    ├── vit_checkpoint/             # best ViT checkpoint + figures
    └── rvit_checkpoint/            # best RViT checkpoint + figures
```

Datasets, `.pt` tensors, and checkpoints are **not** in the repo (too large). Get the raw
images from Kaggle (link below), place them under `DATASET/Training` and `DATASET/Testing`,
then run the data-prep step below.

---

## Setup

```bash
pip install torch torchvision tqdm pillow scikit-learn matplotlib
```

Dataset structure (place the Kaggle images here):

```
DATASET/
├── Training/  {glioma, meningioma, notumor, pituitary}/
└── Testing/   {glioma, meningioma, notumor, pituitary}/
```

### Prepare data (run once, from this directory)

```bash
python DATASET/data_prep.py
```

Reads `DATASET/Training` and `DATASET/Testing`, writes `Training.pt` (5600 images) and
`Testing.pt` (1600 images) into the current directory — 224×224 RGB in `[0, 1]`.
All train/evaluate scripts expect these files in the repo root.

### Note on PDCNN checkpoints

`PDCNN_train.py` (unmodified reference implementation) always saves to `checkpoint/`.
Our evaluator reads `pdcnn_checkpoint/` — after a PDCNN training run, rename the folder:

```powershell
Move-Item checkpoint pdcnn_checkpoint -Force
```

---

## Train

```bash
python PDCNN_train.py            # 50 epochs,  Adam 5e-4,    -> checkpoint/
python VITB32_train.py           # 10 epochs + early stopping (Adam 1e-4, paper head) -> vit_checkpoint/vit_b32_best.pth
python RVIT_FUSIONNET_train.py   # 50 epochs,  AdamW differential LR, progressive unfreezing, cosine anneal -> rvit_checkpoint/rvit_best.pth
```

Quick sanity check: set `SMOKE_TEST = True` at the top of `VITB32_train.py` / `RVIT_FUSIONNET_train.py` (1 epoch on 32 samples).

Each training run prints per-epoch train/val loss+accuracy, a full test report (macro/weighted P/R/F1, per-class table), and saves:
- best checkpoint (on validation-accuracy improvement)
- loss/accuracy curve plot (RViT: stage markers; ViT: early-stop marker)
- confusion-matrix figure

---

## Evaluate

All three evaluators share the same interface:

```bash
# Full test-set report (accuracy, macro/weighted P/R/F1, per-class table,
# confusion matrix text + saved matplotlib figure)
python PDCNN_evaluate.py    --split Testing.pt
python VITB32_evaluate.py   --split Testing.pt
python RVIT_evaluate.py     --split Testing.pt

# Single image prediction (class, confidence, probability bars)
python PDCNN_evaluate.py    path/to/mri.jpg
python VITB32_evaluate.py   path/to/mri.jpg
python RVIT_evaluate.py     path/to/mri.jpg
```

Optional flags (all evaluators):

| Flag | Default | Notes |
|---|---|---|
| `--checkpoint`, `-c` | best checkpoint of that model | see auto-pick below |
| `--batch-size` | 128 (PDCNN/ViT), 32 (RViT) | lower if VRAM-constrained |
| `--no-plot` | off | skip the confusion-matrix figure (headless) |

Checkpoint auto-pick:
- PDCNN: highest `vacc` in `pdcnn_checkpoint/`
- ViT: `vit_checkpoint/vit_b32_best.pth`
- RViT: `rvit_checkpoint/rvit_best.pth`, falling back to `old/rvit_stage2.pth` / `old/rvit_stage1.pth`

The confusion-matrix figure is saved next to the checkpoint, e.g.
`pdcnn_checkpoint\PDCNN_best_ep41_..._Testing_cm.png`.

Evaluators also print model/parameter count, best-epoch training metadata, and split info (shape, value range, class counts).

---

## Checkpoint format

```python
{
    "epoch": int,
    "model_state_dict": dict,
    "train_loss": float,
    "val_loss": float,
    "val_acc": float,       # 0.0 to 1.0
}
```

Load in code:

```python
# PDCNN
from PDCNN_train import PDCNN
ckpt = torch.load("pdcnn_checkpoint/PDCNN_best_*.pt", map_location="cpu", weights_only=True)
model = PDCNN(num_classes=4)
model.load_state_dict(ckpt["model_state_dict"])

# ViT-B/32 (paper head lives in VITB32_train.py)
from VITB32_train import build_vit_b32
ckpt = torch.load("vit_checkpoint/vit_b32_best.pth", map_location="cpu", weights_only=True)
model = build_vit_b32(num_classes=4, pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])

# RViT-FusionNet (model lives in RVIT_FUSIONNET_train.py)
from RVIT_FUSIONNET_train import RViTFusionNet
ckpt = torch.load("rvit_checkpoint/rvit_best.pth", map_location="cpu", weights_only=True)
model = RViTFusionNet(num_classes=4)
model.load_state_dict(ckpt["model_state_dict"])
```

Note: PDCNN consumes raw `[0,1]` inputs; ViT/RViT apply ImageNet normalization inside training and evaluation.

---

## Dataset

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** by Masoud Nickparvar (CC BY 4.0)

7,200 brain MRI images combined from figshare, SARTAJ, and Br35H datasets. Images are resized to 224×224 RGB.

| Class | Train | Test |
|---|---|---|
| Glioma | 1400 | 400 |
| Meningioma | 1400 | 400 |
| No tumor | 1400 | 400 |
| Pituitary | 1400 | 400 |
| **Total** | **5600** | **1600** |
