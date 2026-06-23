# Requires NVIDIA Container Toolkit on the host (run with `--gpus all`).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
# torch ships in the base image; pip treats the pinned torch as already satisfied.
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY static ./static

EXPOSE 8000
# Single worker on purpose: load the model once, avoid OOM on small GPUs.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
