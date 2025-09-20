# src/services/inference.py
from src.models.CT.swin_wrapper import SwinHFClassifier
from src.models.xray.xray_wrapper import XrayClassifier
from .insights import query_gemini_flash
from .label_map import LABEL_REPHRASE

_swin_service = None
_xray_service = None


def run_inference(modality: str, file_path: str):
    global _swin_service, _xray_service

    if modality.lower() == "ct":
        if _swin_service is None:
            _swin_service = SwinHFClassifier(
                model_id="Koushim/breast-cancer-swin-classifier",
                device="cpu"
            )
        result = _swin_service.predict_single(file_path)

    elif modality.lower() == "x-ray":
        if _xray_service is None:
            _xray_service = XrayClassifier(
                model_id="nickmuchi/vit-finetuned-chest-xray-pneumonia",
                labels=["Normal", "Pneumonia"],
                device="cpu"
            )
        result = _xray_service.predict(file_path)

    else:
        result = {
            "label_name": "Not Implemented",
            "confidence": 0.0,
            "ai_insight": "No model integrated yet for this modality."
        }

    # Rephrase labels
    label_name = result.get("label_name", "unknown")
    pretty_label = LABEL_REPHRASE.get(label_name, label_name)
    result["label_name"] = pretty_label

    # Add Gemini insight
    confidence = result.get("confidence", 0.0)
    result["ai_insight"] = query_gemini_flash(modality, pretty_label, float(confidence))

    return result
