import bentoml
from prometheus_client import Counter

from .classifier import YourToxicityClassifier


INFERENCE_COUNT = Counter(
    "app_http_inference_count",
    "Total number of HTTP inference requests",
)


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
        INFERENCE_COUNT.inc()
        return self.model.predict(text)

    @bentoml.api(
        route="/predict_batched",
        batchable=True,
        max_batch_size=64,
        max_latency_ms=1000,
    )
    def predict_batched(self, texts: list[str]) -> list[dict]:
        INFERENCE_COUNT.inc(len(texts))
        return self.model.predict_many(texts)
