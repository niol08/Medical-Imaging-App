
import torch
from typing import Dict, Optional
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class XrayClassifier:
    """
    Wrapper for Hugging Face pneumonia classifier (Chest X-ray).
    Returns a dict with standardized keys.
    """

    def __init__(self,
                 model_id: str = "nickmuchi/vit-finetuned-chest-xray-pneumonia",
                 labels: Optional[list] = None,
                 device: Optional[str] = None):
        self.model_id = model_id
        self.labels = labels or ["Normal", "Pneumonia"]
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device).eval()

    def predict(self, path: str) -> Dict:
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        best_idx = int(probs.argmax())
        return {
            "label_name": self.labels[best_idx],
            "confidence": float(probs[best_idx]),
            "all_scores": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        }
