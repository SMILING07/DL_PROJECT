import os
from PIL import Image
import torch
import random
from tqdm import tqdm
IMAGE_SIZE = (224, 224)
classes = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}
for split in os.listdir("DATASET"):
    split_path = os.path.join("DATASET", split)
    images = []
    labels = []
    for category in tqdm(os.listdir(split_path), colour="red"):
        category_path = os.path.join(split_path, category)
        if category not in classes:
            continue
        label = classes[category]
        for image_name in tqdm(
            os.listdir(category_path),
            colour="green",
            leave=False
        ):
            image_path = os.path.join(category_path, image_name)
            if os.path.isfile(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    img = img.resize(IMAGE_SIZE)
                    img = torch.tensor(
                        list(img.getdata()),
                        dtype=torch.float32
                    )
                    img = img.reshape(
                        IMAGE_SIZE[1],
                        IMAGE_SIZE[0],
                        3
                    )
                    img = img.permute(2, 0, 1)
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error reading {image_path}: {e}")
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = zip(*data)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = {
        "images": images,
        "labels": labels
    }
    torch.save(dataset, split + ".pt")
    print(
        f"{split}: "
        f"images={images.shape}, "
        f"labels={labels.shape}"
    )