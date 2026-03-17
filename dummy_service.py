import bentoml

from src.classifier import YourToxicityClassifier


@bentoml.service(
    image=bentoml.images.Image(python_version="3.11").python_packages(
        "torch", "transformers", "prometheus-client"
    ),
)
class ToxicService:
    def __init__(self) -> None:
        self.model = YourToxicityClassifier()

    @bentoml.api(route="/predict")
    def predict(self, text: str) -> dict:
        return self.model.predict(text)
