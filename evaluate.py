"""
Evaluate a single brain MRI image using a trained PDCNN checkpoint.
Usage:
    python evaluate.py path/to/image.jpg
    python evaluate.py path/to/image.jpg --checkpoint checkpoint/PDCNN_best_ep12_trloss0.3421_vlloss0.2891_vacc92.50.pt
"""
import argparse
import sys
from pathlib import Path
import torch
from PIL import Image
from pdcnn import PDCNN, DEVICE  # import model classes from pdcnn.py

#DO NOT CHANGE THESE
CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "no tumor", 3: "pituitary"}

def load_checkpoint(checkpoint_path: str) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)

def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
    tensor = tensor.reshape(224, 224, 3).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)  # add batch dimension -> (1, 3, 224, 224)

def find_latest_checkpoint(checkpoint_dir: str = "checkpoint") -> str | None:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    
    # prefer best checkpoint, fall back to any
    candidates = sorted(ckpt_dir.glob("PDCNN_best_*.pt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("PDCNN_final_*.pt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("*.pt"))

    return str(candidates[-1]) if candidates else None


def predict(model: PDCNN, image_tensor: torch.Tensor) -> tuple[int, str, torch.Tensor]:
    model.eval()
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    class_idx = predicted.item()
    return class_idx, CLASS_NAMES[class_idx], probs.squeeze()


def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor class from an MRI image.")
    parser.add_argument("image", type=str, help="Path to the MRI image")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str, default=None,
        help="Path to the model checkpoint (default: latest in checkpoint/)",
    )
    args = parser.parse_args()

    # validate image
    if not Path(args.image).is_file():
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    # find checkpoint 
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if ckpt_path is None or not Path(ckpt_path).is_file():
        print("Error: no checkpoint found. Train the model first or provide --checkpoint.")
        sys.exit(1)
    print(f"Checkpoint : {ckpt_path}")

    # load model 
    ckpt = load_checkpoint(ckpt_path)
    model = PDCNN(num_classes=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} — val acc: {ckpt['val_acc'] * 100:.2f}%")
    # preprocess image 
    image_tensor = preprocess_image(args.image)
    # predict 
    class_idx, class_name, probs = predict(model, image_tensor)
    print(f"\nImage     : {args.image}")
    print(f"Prediction: {class_name} (class {class_idx})")
    print(f"Confidence: {probs[class_idx].item() * 100:.2f}%")
    print("\nClass probabilities:")
    for i, name in CLASS_NAMES.items():
        bar = "█" * int(probs[i].item() * 30)
        print(f"  {name:>10s}: {probs[i].item() * 100:6.2f}% {bar}")

if __name__ == "__main__":
    main()