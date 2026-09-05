"""
Hugging Face Dataset Loader and Downloader for Plant Disease Detection
Downloads, streams, and prepares PlantVillage and other plant pathology datasets
directly from Hugging Face Hub (huggingface.co/datasets).
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / os.getenv("DATA_PATH", "data/plantvillage")

# Popular and verified Plant Disease datasets on Hugging Face
RECOMMENDED_HF_DATASETS = {
    "faisal-hugging-face/plant-disease": "Full PlantVillage dataset covering Tomato, Potato, Corn, Apple, Grape, Pepper",
    "flwrlabs/plant-village": "Cleaned PlantVillage benchmark dataset partitioned for federated and centralized learning",
    "beans": "Official lightweight Hugging Face leaf disease dataset (Angular Leaf Spot, Bean Rust, Healthy)",
    "ayerr/plant-disease-classification": "Multi-crop disease classification dataset with high-resolution imagery",
    "Francesco/cotton-plant-disease": "Specialized agricultural disease dataset for cotton leaf foliage"
}


def download_hf_dataset(
    repo_id: str = "faisal-hugging-face/plant-disease",
    output_dir: Path = DATA_DIR,
    max_samples_per_class: Optional[int] = None,
    split_ratio: float = 0.8
) -> Dict[str, int]:
    """
    Downloads a plant disease dataset from Hugging Face and saves it in the standard
    data/plantvillage/{train,val}/class_name/ structure.
    """
    print("=" * 65)
    print(f"  DOWNLOADING HUGGING FACE DATASET: {repo_id}")
    print(f"  Target Directory: {output_dir}")
    print("=" * 65)

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required. Run: pip install datasets")

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HuggingFace] Fetching dataset stream from {repo_id} ...")
    try:
        # Load dataset
        ds = load_dataset(repo_id)
    except Exception as e:
        print(f"[HuggingFace Error] Failed to load dataset {repo_id}: {e}")
        print("[HuggingFace] Falling back to downloading dataset via snapshot_download...")
        try:
            from huggingface_hub import snapshot_download
            local_snapshot = snapshot_download(repo_id=repo_id, repo_type="dataset")
            print(f"[HuggingFace] Dataset snapshot downloaded to {local_snapshot}")
            return {"status": "snapshot_downloaded", "path": str(local_snapshot)}
        except Exception as snap_err:
            print(f"[HuggingFace Error] Snapshot download failed: {snap_err}")
            return {}

    # Identify image and label column names
    first_split = list(ds.keys())[0]
    sample_item = ds[first_split][0]

    image_col = None
    label_col = None

    for col in sample_item.keys():
        if isinstance(sample_item[col], Image.Image) or col.lower() in ["image", "img", "pixel_values"]:
            image_col = col
        if col.lower() in ["label", "labels", "category", "class", "target"]:
            label_col = col

    if not image_col or not label_col:
        print(f"[HuggingFace] Unrecognized column structure: {list(sample_item.keys())}")
        return {}

    # Get class names from features if ClassLabel
    features = ds[first_split].features
    if hasattr(features[label_col], "names"):
        class_names = features[label_col].names
    else:
        # Collect distinct string labels
        unique_labels = sorted(list(set(ds[first_split][label_col])))
        class_names = [str(l) for l in unique_labels]

    print(f"[HuggingFace] Found {len(class_names)} classes: {class_names[:5]}...")

    class_counters = {c: 0 for c in class_names}
    exported_counts = {"train": 0, "val": 0}

    # Iterate splits or split automatically
    all_data = []
    for split_name in ds.keys():
        for item in ds[split_name]:
            all_data.append(item)

    import random
    random.seed(42)
    random.shuffle(all_data)

    print(f"[HuggingFace] Processing {len(all_data)} items...")

    for idx, item in enumerate(all_data):
        raw_label = item[label_col]
        class_label = class_names[raw_label] if isinstance(raw_label, int) and raw_label < len(class_names) else str(raw_label)
        
        # Clean label for folder name
        class_label_clean = class_label.replace(" ", "_").replace(":", "_").replace("/", "_")

        if max_samples_per_class and class_counters.get(class_label_clean, 0) >= max_samples_per_class:
            continue

        class_counters[class_label_clean] = class_counters.get(class_label_clean, 0) + 1
        current_count = class_counters[class_label_clean]

        # Determine split
        split_target = "train" if (current_count % 5 != 0) else "val"
        target_folder = (output_dir / split_target) / class_label_clean
        target_folder.mkdir(parents=True, exist_ok=True)

        img = item[image_col]
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(img).convert("RGB")
            except Exception:
                continue
        else:
            img = img.convert("RGB")

        dest_path = target_folder / f"hf_{idx:05d}.jpg"
        img.save(dest_path, "JPEG", quality=92)
        exported_counts[split_target] += 1

        if (idx + 1) % 250 == 0:
            print(f"  -> Processed {idx + 1}/{len(all_data)} images...")

    print("=" * 65)
    print(f" HUGGING FACE DATASET PREPARATION COMPLETE!")
    print(f" Train Images Exported: {exported_counts['train']}")
    print(f" Val Images Exported:   {exported_counts['val']}")
    print(f" Target Directory:      {output_dir}")
    print("=" * 65)

    return exported_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Plant Disease Dataset from Hugging Face")
    parser.add_argument("--repo", type=str, default="faisal-hugging-face/plant-disease",
                        help="Hugging Face dataset repository ID")
    parser.add_argument("--output", type=str, default=str(DATA_DIR),
                        help="Output directory for dataset")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Maximum samples per class (useful for quick testing)")
    parser.add_argument("--list-popular", action="store_true",
                        help="List recommended Hugging Face datasets")
    args = parser.parse_args()

    if args.list_popular:
        print("\nRecommended Hugging Face Plant Disease Datasets:")
        for k, v in RECOMMENDED_HF_DATASETS.items():
            print(f" • {k.ljust(35)} : {v}")
    else:
        download_hf_dataset(
            repo_id=args.repo,
            output_dir=Path(args.output),
            max_samples_per_class=args.max_per_class
        )
