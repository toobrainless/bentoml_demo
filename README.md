# BentoML Demo

A small demo showing how to wrap an ML model into an API service with BentoML and package it for deployment.

### What is BentoML?

BentoML is a framework for model serving. It helps turn a Python model or inference script into a service with minimal boilerplate.

Workflow:

`service → serve → build → containerize`

### Why use it?

- expose a model through an HTTP API
- run it locally
- package dependencies with the service
- build a deployable artifact
- containerize it without writing much backend code
- adaptive batching
- monitoring and metrics

#### In result

- better CPU/GPU utilization
- production-oriented serving workflow

### What is a Bento artifact?

A **Bento** is a versioned deployable package of a BentoML service.

It includes:

- service code
- Python dependencies
- model artifacts
- runtime configuration

## Demo Guide

#### Structure

- `src/classifier.py` — model wrapper with plain Python inference methods
- `src/service.py` — BentoML service layer exposing the model as an API
- `src/benchmark.py` — small benchmark comparing plain requests and batched requests

#### Installation

```bash
uv sync
```

#### Run the service

```bash
uv run bentoml serve src.service:ToxicService
```

#### Build a Bento

```bash
uv run bentoml build
uv run bentoml list
```

#### Containerize

```bash
uv run bentoml containerize <bento_name:version>
docker run --rm -p 3000:3000 <bento_name:version>
```

#### Deploy

```bash
uv run bentoml deploy .
# or
uv run bentoml deploy <bento_name:version> -n toxic-demo
```

#### Test request

```bash
curl -X POST http://localhost:3000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"thank you very much"}'
```

#### Metrics

The service exposes a simple Prometheus metric:

* `app_http_inference_count` — total number of inference requests

#### Benchmark

Run:

```bash
python3 src/benchmark.py
```

It compares:

* `/predict` — plain inference
* `/predict_batched` — BentoML endpoint with adaptive batching

## References

1. [https://docs.bentoml.com/en/latest/get-started/hello-world.html](https://docs.bentoml.com/en/latest/get-started/hello-world.html)
2. [https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html)
3. [https://docs.bentoml.com/en/latest/get-started/adaptive-batching.html](https://docs.bentoml.com/en/latest/get-started/adaptive-batching.html)
4. [https://bentoml.com/blog/breaking-up-with-flask-amp-fastapi-why-ml-model-serving-requires-a-specialized-framework](https://bentoml.com/blog/breaking-up-with-flask-amp-fastapi-why-ml-model-serving-requires-a-specialized-framework)

