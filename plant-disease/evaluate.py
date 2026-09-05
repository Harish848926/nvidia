"""
Model Evaluation and Performance Validation Module
Computes overall accuracy, Top-3 accuracy, per-class F1-scores, and confusion matrix.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv

load_dotenv()

from train import (
    CLASSES,
    IDX_TO_CLASS,
    MODEL_PATH,
    DATA_DIR,
    PlantVillageDataset,
    get_transforms,
    build_vision_model,
    prepare_starter_data
)


def evaluate_model(
    model_path: Path = MODEL_PATH,
    data_dir: Path = DATA_DIR,
    batch_size: int = 16
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  PLANT DISEASE MODEL EVALUATION & VALIDATION")
    print(f"  Device: {device} | Model Checkpoint: {model_path}")
    print("=" * 65)

    if not data_dir.exists():
        prepare_starter_data(data_dir)

    val_dir = data_dir / "val"
    if not val_dir.exists() or len(list(val_dir.glob("*/*"))) == 0:
        val_dir = data_dir / "train"

    _, val_transform = get_transforms()
    val_dataset = PlantVillageDataset(val_dir, transform=val_transform)

    if len(val_dataset) == 0:
        print("[Evaluate Error] No evaluation samples found.")
        return {}

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_vision_model(num_classes=len(CLASSES)).to(device)
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"[Evaluate] Loaded weights from {model_path}")
    else:
        print(f"[Evaluate Warning] No checkpoint at {model_path}. Evaluating untrained initialized model.")

    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    top3_correct = 0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

            # Top 1
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Top 3
            _, top3 = outputs.topk(min(3, len(CLASSES)), dim=1)
            top3_correct += top3.eq(targets.view(-1, 1).expand_as(top3)).sum().item()

    acc = (sum(p == t for p, t in zip(all_preds, all_targets)) / total_samples) * 100.0
    top3_acc = (top3_correct / total_samples) * 100.0
    avg_loss = total_loss / total_samples

    present_classes = sorted(list(set(all_targets)))
    target_names = [CLASSES[i] for i in present_classes]

    report_str = classification_report(all_targets, all_preds, labels=present_classes, target_names=target_names, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=present_classes)

    print("\n" + "-" * 65)
    print(f" Overall Accuracy:     {acc:.2f}%")
    print(f" Top-3 Accuracy:       {top3_acc:.2f}%")
    print(f" Average Cross-Entropy:{avg_loss:.4f}")
    print(f" Total Samples Tested: {total_samples}")
    print("-" * 65)
    print("\nDetailed Per-Class Classification Report:\n")
    print(report_str)

    report_file = model_path.parent / "eval_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        f.write(f"Accuracy: {acc:.2f}%\nTop-3 Accuracy: {top3_acc:.2f}%\nLoss: {avg_loss:.4f}\n\n{report_str}")
    print(f"[Evaluate] Performance summary saved to {report_file}")

    return {
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "loss": avg_loss,
        "report": report_str
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Model")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Model checkpoint path")
    parser.add_argument("--data", type=str, default=str(DATA_DIR), help="Dataset root directory")
    args = parser.parse_args()

    evaluate_model(model_path=Path(args.model), data_dir=Path(args.data))
