# Requires NVIDIA Container Toolkit on the host (run with `--gpus all`).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
# torch already ships in the base image; install the rest.
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" "pydantic>=2.7" \
    "diffusers>=0.32" "transformers>=4.46" "accelerate>=1.0"

COPY server.py .
COPY static ./static

EXPOSE 8000
# Single worker on purpose: load the model once, avoid OOM on small GPUs.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
