# PDCNN — Brain Tumor Classification

Parallel Dilated CNN that classifies brain MRI scans into 4 classes: **glioma**, **meningioma**, **no tumor**, **pituitary**.

**Best validation accuracy: 95.06%**

---

## Architecture

```
Input (3, 224, 224)
        │
     ┌──┴──┐
     │ Stem │  Conv→BN→ReLU → Conv→BN→ReLU → MaxPool
     └──┬──┘
        │
   ┌────┼────┬────┐
   │    │    │    │
  d=1  d=2  d=3   │   ← 3 parallel dilated branches
   │    │    │    │     (local / medium / large features)
   └────┼────┼────┘
        │ concat (1536)
     ┌──┴──┐
     │ FC  │  Linear→BN→ReLU→Drop(0.5)→Linear→ReLU→Drop(0.3)→Linear
     └──┬──┘
        │
   Output (4)
```

| Component | Details |
|---|---|
| Stem | 2 × Conv(3→64) + MaxPool → 112×112 |
| Branches | 3 parallel, each: Conv(64→128→256→512) + GAP |
| Dilation rates | 1 (local), 2 (medium), 3 (large) |
| Classifier | 1536 → 512 → 128 → 4 |
| Parameters | ~5.5M |

---

## Setup

```bash
pip install torch torchvision tqdm pillow
```

Place your dataset in this structure:
```
DATASET/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## Run

### 1. Prepare data

Converts raw images to `.pt` tensors (run once).

```bash
python data_prep.py
```

Outputs `Training.pt` and `Testing.pt`.

### 2. Train

```bash
python pdcnn.py
```

Checkpoints saved to `checkpoint/`:
- `PDCNN_best_*.pt` — saved when validation accuracy improves
- `PDCNN_final_*.pt` — saved at end of training

**Config** (edit in `pdcnn.py`):

| Parameter | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | 32 | Lower if VRAM < 8 GB |
| `EPOCHS` | 50 | |
| `LR` | 5e-4 | |
| `NUM_CLASSES` | 4 | Don't change |

### 3. Evaluate

```bash
python evaluate.py path/to/mri.jpg
```

Auto-loads latest checkpoint. Or specify one:

```bash
python evaluate.py path/to/mri.jpg --checkpoint checkpoint/PDCNN_best_ep12_trloss0.3421_vlloss0.2891_vacc92.50.pt
```

Output:
```
Prediction: glioma (class 0)
Confidence: 94.12%

Class probabilities:
      glioma:  94.12% ████████████████████████████
   meningioma:   2.31% █
     no tumor:   1.05%
    pituitary:   2.52% █
```

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
from pdcnn import PDCNN

ckpt = torch.load("checkpoint/PDCNN_best_*.pt", map_location="cpu")
model = PDCNN(num_classes=4)
model.load_state_dict(ckpt["model_state_dict"])
```

---

## Dataset

Brain MRI images (224×224 RGB) from Kaggle, 4 balanced classes:

| Class | Train | Test |
|---|---|---|
| Glioma | 1400 | 400 |
| Meningioma | 1400 | 400 |
| No tumor | 1400 | 400 |
| Pituitary | 1400 | 400 |
| **Total** | **5600** | **1600** |
