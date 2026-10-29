"""
Evaluate PDCNN on a single MRI image OR a full .pt test split.
Usage:
    python PDCNN_evaluate.py path/to/image.jpg
    python PDCNN_evaluate.py path/to/image.jpg --checkpoint pdcnn_checkpoint/PDCNN_best_ep41_trloss0.3617_vlloss0.2223_vacc95.06.pt
    python PDCNN_evaluate.py --split Testing.pt
    python PDCNN_evaluate.py --split Testing.pt --checkpoint pdcnn_checkpoint/PDCNN_best_ep41_trloss0.3617_vlloss0.2223_vacc95.06.pt
    (a positional *.pt file also works: python PDCNN_evaluate.py Testing.pt)
"""
import argparse
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from PDCNN_train import PDCNN, DEVICE  # import model classes from PDCNN_train.py

#DO NOT CHANGE THESE
CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "no tumor", 3: "pituitary"}
MODEL_NAME = "PDCNN"

def load_checkpoint(checkpoint_path: str) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)

def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
    tensor = tensor.reshape(224, 224, 3).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)  # add batch dimension -> (1, 3, 224, 224)

def find_latest_checkpoint(checkpoint_dir: str = "pdcnn_checkpoint") -> str | None:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    
    # prefer checkpoint with highest val accuracy (vacc in filename),
    # since alphabetical order mis-sorts epochs (ep8 > ep41 as strings)
    def best_key(p: Path) -> float:
        m = re.search(r"vacc(\d+(?:\.\d+)?)", p.name)
        return float(m.group(1)) if m else -1.0

    candidates = sorted(ckpt_dir.glob("PDCNN_best_*.pt"), key=best_key, reverse=True)
    if not candidates:
        candidates = sorted(ckpt_dir.glob("PDCNN_final_*.pt"), key=best_key, reverse=True)
    if not candidates:
        candidates = sorted(ckpt_dir.glob("*.pt"), key=best_key, reverse=True)

    return str(candidates[0]) if candidates else None


def predict(model: PDCNN, image_tensor: torch.Tensor) -> tuple[int, str, torch.Tensor]:
    model.eval()
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    class_idx = predicted.item()
    return class_idx, CLASS_NAMES[class_idx], probs.squeeze()


@torch.no_grad()
def evaluate_dataset(model: PDCNN, split_path: str, batch_size: int = 128) -> dict:
    """Run batched inference over a saved split and return full metrics."""
    data = torch.load(split_path, map_location="cpu", weights_only=True)
    x, y = data["images"], data["labels"]
    n, num_classes = x.shape[0], len(CLASS_NAMES)
    model.eval()
    # confusion matrix: rows = true, cols = predicted
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size].to(DEVICE, non_blocking=True)
        preds = model(xb).argmax(dim=1).cpu()
        for t, p in zip(y[i:i + batch_size], preds):
            conf[int(t), int(p)] += 1
        del xb
    # per-class counts
    per_class, total_correct = {}, int(conf.diag().sum().item())
    for c in range(num_classes):
        tp = int(conf[c, c].item())
        fp = int(conf[:, c].sum().item()) - tp
        fn = int(conf[c, :].sum().item()) - tp
        support = int(conf[c, :].sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {
            "name": CLASS_NAMES[c], "precision": precision, "recall": recall,
            "f1": f1, "support": support, "correct": tp,
            "class_accuracy": tp / support if support else 0.0,
        }
    accuracy = total_correct / n if n else 0.0
    macro = {
        k: sum(v[k] for v in per_class.values()) / num_classes
        for k in ("precision", "recall", "f1", "class_accuracy")
    }
    # support-weighted averages (sklearn "weighted avg")
    weighted = {
        k: (sum(v[k] * v["support"] for v in per_class.values()) / n) if n else 0.0
        for k in ("precision", "recall", "f1")
    }
    split_info = {
        "shape": tuple(x.shape), "dtype": str(x.dtype),
        "min": float(x.min()), "max": float(x.max()),
    }
    return {
        "n": n, "confusion": conf, "per_class": per_class,
        "accuracy": accuracy, "macro": macro, "weighted": weighted,
        "correct": total_correct, "split_info": split_info,
    }


def print_report(report: dict, split_path: str) -> None:
    n, acc, si = report["n"], report["accuracy"], report["split_info"]
    print(f"\nSplit      : {split_path}  (n={n})")
    print(f"Images     : shape {si['shape']} ({si['dtype']}), "
          f"value range [{si['min']:.2f}, {si['max']:.2f}]")
    print("Classes    : " + ", ".join(
        f"{report['per_class'][c]['name']}={report['per_class'][c]['support']}"
        for c in sorted(report["per_class"])))
    print(f"Accuracy   : {acc * 100:.2f}%  ({report['correct']}/{n} correct)")
    print(f"Macro avg  : P={report['macro']['precision'] * 100:.2f}%  "
          f"R={report['macro']['recall'] * 100:.2f}%  "
          f"F1={report['macro']['f1'] * 100:.2f}%")
    print(f"Weighted avg: P={report['weighted']['precision'] * 100:.2f}%  "
          f"R={report['weighted']['recall'] * 100:.2f}%  "
          f"F1={report['weighted']['f1'] * 100:.2f}%")
    print("\nPer-class metrics:")
    print(f"  {'class':>10s} | {'acc':>7s} | {'prec':>7s} | {'recall':>7s} | {'f1':>7s} | support")
    print("  " + "-" * 62)
    for c in sorted(report["per_class"]):
        v = report["per_class"][c]
        print(f"  {v['name']:>10s} | {v['class_accuracy'] * 100:6.2f}% | "
              f"{v['precision'] * 100:6.2f}% | {v['recall'] * 100:6.2f}% | "
              f"{v['f1'] * 100:6.2f}% | {v['support']}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "  true\\pred |" + "".join(f"{CLASS_NAMES[c][:6]:>8s}" for c in sorted(CLASS_NAMES))
    print(header)
    for r in sorted(report["per_class"]):
        row = report["confusion"][r].tolist()
        print(f"  {CLASS_NAMES[r]:>10s} |" + "".join(f"{v:>8d}" for v in row))


def print_training_info(ckpt: dict, model, model_name: str = MODEL_NAME) -> None:
    params = sum(p.numel() for p in model.parameters())
    print(f"Model      : {model_name} ({params:,} parameters, device={DEVICE})")
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        parts = [f"best epoch {ckpt['epoch']}"]
        if "val_acc" in ckpt:
            parts.append(f"val acc {float(ckpt['val_acc']) * 100:.2f}%")
        if "val_loss" in ckpt:
            parts.append(f"val loss {float(ckpt['val_loss']):.4f}")
        if "train_loss" in ckpt:
            parts.append(f"train loss {float(ckpt['train_loss']):.4f}")
        print("Trained    : " + ", ".join(parts))
    else:
        print("Trained    : (no training metadata in this checkpoint)")


def plot_confusion_matrix(report: dict, title: str, save_path: Path, show: bool = True) -> None:
    labels = [report["per_class"][c]["name"] for c in sorted(CLASS_NAMES)]
    ConfusionMatrixDisplay(report["confusion"], display_labels=labels).plot(
        cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure     : {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor class from an MRI image or evaluate a .pt split.")
    parser.add_argument("image", type=str, nargs="?", default=None,
                        help="Path to the MRI image (or a .pt split, same as --split)")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str, default=None,
        help="Path to the model checkpoint (default: latest in pdcnn_checkpoint/)",
    )
    parser.add_argument(
        "--split", "-s", type=str, default=None,
        help="Path to a saved split (.pt with images+labels) for full metrics, e.g. Testing.pt",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for --split evaluation")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the confusion-matrix figure (for headless runs)")
    args = parser.parse_args()

    # --split mode (or positional .pt file)
    split_path = args.split
    if split_path is None and args.image is not None and args.image.lower().endswith(".pt"):
        split_path = args.image

    # find checkpoint 
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if ckpt_path is None or not Path(ckpt_path).is_file():
        print("Error: no checkpoint found. Train the model first or provide --checkpoint.")
        sys.exit(1)
    print(f"Checkpoint : {ckpt_path}")

    # load model 
    ckpt = load_checkpoint(ckpt_path)
    model = PDCNN(num_classes=4)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print_training_info(ckpt, model)

    # dataset mode: full precision/recall/F1/accuracy report + figure
    if split_path is not None:
        if not Path(split_path).is_file():
            print(f"Error: split not found: {split_path}")
            sys.exit(1)
        report = evaluate_dataset(model, split_path, batch_size=args.batch_size)
        print_report(report, split_path)
        if not args.no_plot:
            save_path = Path(ckpt_path).parent / f"{Path(ckpt_path).stem}_{Path(split_path).stem}_cm.png"
            plot_confusion_matrix(report, f"{MODEL_NAME} — {Path(split_path).name}", save_path)
        return

    # single-image mode
    if args.image is None or not Path(args.image).is_file():
        print("Error: provide an image path or --split <file.pt>")
        sys.exit(1)
    # preprocess image 
    image_tensor = preprocess_image(args.image)
    # predict 
    class_idx, class_name, probs = predict(model, image_tensor)
    print(f"\nImage     : {args.image}")
    print(f"Prediction: {class_name} (class {class_idx})")
    print(f"Confidence: {probs[class_idx].item() * 100:.2f}%")
    print("\nClass probabilities:")
    for i, name in CLASS_NAMES.items():
        bar = "#" * int(probs[i].item() * 30)
        print(f"  {name:>10s}: {probs[i].item() * 100:6.2f}% {bar}")

if __name__ == "__main__":
    main()