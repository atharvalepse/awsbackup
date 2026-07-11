"""FastAPI app exposing ``/metrics`` (Prometheus text format) and ``/health``.

Run with:

    uvicorn orchestration.observability.metrics_endpoint:app --port 9090
"""
from fastapi import FastAPI
from fastapi.responses import Response

from orchestration.observability.metrics import CounterRegistry


app = FastAPI(title="GML Orchestration Metrics")


@app.get("/metrics")
def metrics() -> Response:
    text = CounterRegistry.get().render_prometheus()
    return Response(content=text, media_type="text/plain; version=0.0.4")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
