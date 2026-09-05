"""
Training Module for Plant Disease Vision Model
Supports PlantVillage dataset with NVIDIA DGX GPU/AMP optimization.
Saves model checkpoint to model/plant_model.pth
"""

import os
import sys
import time
import math
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / os.getenv("DATA_PATH", "data/plantvillage")
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "model/plant_model.pth")

# Disease classes covering Tomato, Potato, and Rice
CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Leaf_Blast",
    "Rice___Bacterial_blight",
    "Rice___Brown_spot",
    "Rice___healthy"
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    return train_transform, val_transform


class PlantVillageDataset(Dataset):
    """
    Standard PyTorch dataset loader for PlantVillage directory structure:
    data/plantvillage/{train,val}/class_name/xxx.jpg
    """
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if self.root_dir.exists():
            for class_dir in sorted(self.root_dir.iterdir()):
                if class_dir.is_dir() and class_dir.name in CLASS_TO_IDX:
                    label = CLASS_TO_IDX[class_dir.name]
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in valid_exts:
                            self.samples.append((str(img_file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def synthesize_leaf(class_name: str, size: int = 256) -> Image.Image:
    """Generates synthetic leaf texture with class-specific disease signatures."""
    bg_color = (random.randint(230, 245), random.randint(225, 240), random.randint(220, 235))
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    is_rice = "Rice" in class_name
    is_healthy = "healthy" in class_name

    leaf_col = (40, 160, 50) if is_healthy else (75, 130, 45)

    if is_rice:
        # Narrow blade
        pts = [(cx - 20, cy - 100), (cx + 20, cy - 100), (cx + 15, cy + 100), (cx - 15, cy + 100)]
        draw.polygon(pts, fill=leaf_col, outline=(30, 100, 30))
    else:
        # Broad leaf
        pts = []
        for i in range(40):
            a = i * (2 * math.pi / 40)
            r = 0.35 * size * (1.0 + 0.1 * math.sin(a * 4))
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, fill=leaf_col, outline=(25, 85, 25))

    # Veins
    draw.line([(cx, cy - 80), (cx, cy + 80)], fill=(130, 190, 100), width=3)

    # Pathology spots
    if "Early_blight" in class_name:
        for _ in range(4):
            sx, sy = cx + random.randint(-30, 30), cy + random.randint(-30, 30)
            draw.ellipse([sx - 14, sy - 14, sx + 14, sy + 14], fill=(180, 170, 40))
            draw.ellipse([sx - 9, sy - 9, sx + 9, sy + 9], fill=(80, 50, 25))
            draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(40, 25, 15))
    elif "Late_blight" in class_name:
        for _ in range(3):
            sx, sy = cx + random.randint(-25, 25), cy + random.randint(-35, 35)
            draw.polygon([(sx - 20, sy - 15), (sx + 20, sy - 10), (sx + 15, sy + 20), (sx - 15, sy + 15)],
                         fill=(55, 40, 35), outline=(160, 160, 140))
    elif "Bacterial" in class_name:
        for _ in range(15):
            sx, sy = cx + random.randint(-35, 35), cy + random.randint(-40, 40)
            draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(25, 20, 15))
    elif "Blast" in class_name or "Brown_spot" in class_name:
        for _ in range(6):
            sx, sy = cx + random.randint(-10, 10), cy + random.randint(-60, 60)
            draw.ellipse([sx - 6, sy - 15, sx + 6, sy + 15], fill=(130, 60, 20), outline=(200, 180, 50))

    return img.filter(ImageFilter.GaussianBlur(0.6))


def prepare_starter_data(data_dir: Path):
    """Creates balanced starter images so model training can start immediately."""
    print(f"[Data] Checking dataset in {data_dir}...")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or len(list(train_dir.glob("*/*"))) == 0:
        print("[Data] Generating starter PlantVillage sample images...")
        for split_dir, count in [(train_dir, 12), (val_dir, 4)]:
            for c in CLASSES:
                cdir = split_dir / c
                cdir.mkdir(parents=True, exist_ok=True)
                for i in range(count):
                    img = synthesize_leaf(c)
                    img.save(cdir / f"leaf_{i:03d}.jpg", "JPEG", quality=90)
        print("[Data] Starter dataset generation complete.")


def build_vision_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Builds MobileNetV3 with custom classification head."""
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)

    in_feat = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(256, num_classes)
    )
    return model


def train(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    data_dir: Path = DATA_DIR,
    model_path: Path = MODEL_PATH,
    use_amp: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  PLANT DISEASE CLASSIFIER - TRAINING ENGINE")
    print(f"  Target Device: {device} (CUDA={torch.cuda.is_available()})")
    print(f"  Target Checkpoint: {model_path}")
    print("=" * 65)

    prepare_starter_data(data_dir)

    train_t, val_t = get_transforms()
    train_ds = PlantVillageDataset(data_dir / "train", transform=train_t)
    val_ds = PlantVillageDataset(data_dir / "val", transform=val_t)

    if len(val_ds) == 0:
        val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_vision_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(use_amp and device.type == "cuda"))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = (correct / max(1, total)) * 100.0
        train_loss /= max(1, total)

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * targets.size(0)
                _, preds = outputs.max(1)
                v_corr += preds.eq(targets).sum().item()
                v_tot += targets.size(0)

        val_acc = (v_corr / max(1, v_tot)) * 100.0
        val_loss = v_loss / max(1, v_tot)
        dt = time.time() - t0

        print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | Time: {dt:.1f}s")

        if val_acc >= best_acc or epoch == 1:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "val_acc": val_acc
            }, model_path)

    print(f"\n[Training Finished] Model saved to {model_path} with Best Val Acc: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plant Disease Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="Hugging Face dataset repository ID to download and train on (e.g. faisal-hugging-face/plant-disease)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples per class from Hugging Face dataset")
    args = parser.parse_args()

    if args.hf_dataset:
        print(f"[Train] Downloading/Preparing Hugging Face dataset: {args.hf_dataset} ...")
        from hf_dataset import download_hf_dataset
        download_hf_dataset(repo_id=args.hf_dataset, output_dir=DATA_DIR, max_samples_per_class=args.max_samples)

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, use_amp=not args.no_amp)
