import bentoml
from prometheus_client import Counter

from src.classifier import YourToxicityClassifier


CPU_IMAGE = bentoml.images.Image(
    python_version="3.10",
    lock_python_packages=False,
    python_requirements="""
--index-url https://download.pytorch.org/whl/cpu
--extra-index-url https://pypi.org/simple

torch==2.10.0
transformers==5.3.0
prometheus-client==0.24.1
requests==2.32.5

""",
)


INFERENCE_COUNT = Counter(
    "app_http_inference_count",
    "Total number of HTTP inference requests",
)


@bentoml.service(image=CPU_IMAGE)
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
