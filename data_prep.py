"""
Data Preparation Script

Reads MRI images from DATASET/{split}/{class}/ and converts them into
a single .pt file per split containing:
    - images: (N, 3, 224, 224) float32, normalised to [0, 1]
    - labels: (N,) long tensor
"""

import os
import random

import torch
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = (224, 224)

CLASS_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}


def load_and_convert(split_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []

    for class_name in tqdm(os.listdir(split_path), desc="Classes", colour="red"):
        class_dir = os.path.join(split_path, class_name)
        if class_name not in CLASS_MAP or not os.path.isdir(class_dir):
            continue

        label = CLASS_MAP[class_name]
        for img_name in tqdm(os.listdir(class_dir), desc=class_name, leave=False, colour="green"):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                img = torch.tensor(list(img.getdata()), dtype=torch.float32)
                img = img.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
                img = img.permute(2, 0, 1) / 255.0
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")

    # shuffle
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = zip(*data)

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


if __name__ == "__main__":
    for split in os.listdir("DATASET"):
        split_path = os.path.join("DATASET", split)
        if not os.path.isdir(split_path):
            continue

        print(f"\nProcessing {split}...")
        images, labels = load_and_convert(split_path)
        torch.save({"images": images, "labels": labels}, f"{split}.pt")
        print(f"  Saved {split}.pt — images={images.shape}, labels={labels.shape}")
