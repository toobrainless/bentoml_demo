import torch
from transformers import pipeline


class YourToxicityClassifier:
    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=device,
        )

    def _postprocess(self, result: dict) -> dict:
        label = result["label"].lower()
        score = float(result["score"])
        is_toxic = ("toxic" in label) and (score > 0.5)
        return {
            "is_toxic": is_toxic,
            "classification_label": result["label"],
            "score": score,
        }

    def predict(self, text: str) -> dict:
        result = self.classifier(text)[0]
        return self._postprocess(result)

    def predict_many(self, texts: list[str]) -> list[dict]:
        results = self.classifier(texts)
        return [self._postprocess(result) for result in results]
