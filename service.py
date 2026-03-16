# service.py
import bentoml
import torch
from transformers import pipeline


@bentoml.service(
    image=bentoml.images.Image(python_version="3.11").python_packages(
        "torch", "transformers"
    ),
)
class ToxicService:
    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=device,
        )

    @bentoml.api
    def predict(self, text: str) -> dict:
        result = self.classifier(text)[0]
        label = result["label"].lower()
        score = float(result["score"])

        is_toxic = ("toxic" in label) and (score > 0.5)
        return {
            "is_toxic": is_toxic,
            "label": result["label"],
            "score": score,
        }
