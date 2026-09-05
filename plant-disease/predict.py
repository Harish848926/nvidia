"""
Inference Module for Plant Disease Detection
Integrates PyTorch vision model with RAG knowledge retrieval for automated pathology advisory.
"""

import io
import time
import argparse
from pathlib import Path
from typing import Union, Dict, Any, List

from PIL import Image
import torch
from dotenv import load_dotenv

load_dotenv()

from train import (
    CLASSES,
    IDX_TO_CLASS,
    MODEL_PATH,
    NORM_MEAN,
    NORM_STD,
    build_vision_model,
    get_transforms
)
from rag import get_rag


class PlantPredictor:
    """
    Unified predictor combining Deep Learning vision diagnosis with RAG knowledge retrieval.
    """
    def __init__(self, model_path: Union[str, Path] = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.model = build_vision_model(num_classes=len(CLASSES))

        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            print(f"[Predictor] Loaded model checkpoint from {self.model_path}")
        else:
            print(f"[Predictor Warning] Checkpoint {self.model_path} not found. Running initialized backbone.")

        self.model.to(self.device)
        self.model.eval()

        _, self.val_transform = get_transforms()
        self.rag = get_rag()

    def preprocess(self, image_input: Union[str, Path, bytes, Image.Image]) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        return self.val_transform(img).unsqueeze(0).to(self.device)

    def predict(self, image_input: Union[str, Path, bytes, Image.Image], top_k: int = 3) -> Dict[str, Any]:
        tensor = self.preprocess(image_input)

        start_time = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(CLASSES)))

        top_candidates: List[Dict[str, Any]] = []
        for p, idx in zip(top_probs, top_indices):
            class_name = IDX_TO_CLASS[idx.item()]
            crop, disease = class_name.split("___")
            disease_clean = disease.replace("_", " ")

            top_candidates.append({
                "class_name": class_name,
                "crop": crop,
                "disease": disease_clean,
                "confidence": round(p.item() * 100.0, 2)
            })

        primary = top_candidates[0]

        # Retrieve RAG pathology & treatment knowledge from documents/
        rag_profile = self.rag.get_disease_profile(crop=primary["crop"], disease=primary["disease"])

        return {
            "primary": primary,
            "top_candidates": top_candidates,
            "rag_knowledge": rag_profile,
            "latency_ms": round(latency_ms, 2),
            "device": str(self.device)
        }


_predictor = None

def get_predictor() -> PlantPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PlantPredictor()
    return _predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Plant Disease from Leaf Image")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    args = parser.parse_args()

    predictor = PlantPredictor()
    result = predictor.predict(args.image)

    p = result["primary"]
    rag = result["rag_knowledge"]
    print("\n" + "=" * 60)
    print(f"  PLANT PATHOLOGY DIAGNOSIS REPORT")
    print("=" * 60)
    print(f" Crop:           {p['crop']}")
    print(f" Diagnosis:      {p['disease']}")
    print(f" Confidence:     {p['confidence']}%")
    print(f" Latency:        {result['latency_ms']} ms ({result['device']})")
    print("-" * 60)
    print(" Symptoms (from Agronomy Documents):")
    for s in rag.get("symptoms", [])[:3]:
        print(f"  • {s}")
    print("\n Organic Treatments:")
    for o in rag.get("organic_treatments", [])[:3]:
        print(f"  • {o}")
    print("\n Chemical Treatments:")
    for c in rag.get("chemical_treatments", [])[:3]:
        print(f"  • {c}")
    print("=" * 60)
