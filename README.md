# BentoML Demo

A small demo showing how to wrap an ML model into an API service with BentoML and package it for deployment.

## What is BentoML?

BentoML is a framework for model serving. It helps turn a Python model or inference script into a production-style service without having to manually glue together FastAPI/Flask, Docker, and deployment setup.

In short, the workflow looks like this:

`service → serve → build → containerize`

## Why use it?

### For beginners

BentoML is useful if you want to:

- expose a model through an HTTP API
- run it locally with minimal boilerplate
- package dependencies together with the service
- build a deployable artifact
- containerize it without writing a lot of backend code

This is especially nice if you do not want to spend much time on web frameworks or Docker setup.

### For more advanced users

BentoML also includes features that are more relevant for real model serving workloads, such as:

- dynamic / adaptive batching
- better CPU/GPU utilization
- production-oriented serving workflow
- deployment-friendly packaging
- monitoring and service-level tooling

So it is not just “a quick API wrapper”, but a more specialized framework for ML inference services.

## Notes

For a very small project, a hand-written Dockerfile can still be simpler or even faster to build.  
This demo is more about showing the BentoML workflow and its serving-oriented features.

## References

1. https://docs.bentoml.com/en/latest/get-started/hello-world.html
2. https://bentoml.com/blog/breaking-up-with-flask-amp-fastapi-why-ml-model-serving-requires-a-specialized-framework