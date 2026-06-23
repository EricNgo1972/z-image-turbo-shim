# OpenAI-compatible image-generation shim for Tongyi-MAI/Z-Image-Turbo.
#
# Exposes POST /v1/images/generations matching the OpenAI Images API, so any
# OpenAI/Azure SDK works by swapping the base URL and model name.
import asyncio
import base64
import io
import os
import time
import uuid

import torch
from diffusers import ZImagePipeline
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---- config (via env) ----
MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
API_KEY = os.getenv("API_KEY")  # if set, require "Authorization: Bearer <API_KEY>"
PUBLIC_BASE = os.getenv("PUBLIC_BASE", "http://localhost:8000")  # for response_format=url
DTYPE = os.getenv("DTYPE", "bfloat16")  # "bfloat16" | "float16"
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "1") == "1"  # keep VRAM low (~10GB cards)
IMG_DIR = os.getenv("IMG_DIR", "images")
IMG_TTL_SECONDS = int(os.getenv("IMG_TTL_SECONDS", "3600"))  # url-mode cleanup age

os.makedirs(IMG_DIR, exist_ok=True)

# ---- load model ONCE at startup ----
_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=_dtype, low_cpu_mem_usage=False)
if CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()  # streams weights to GPU as needed -> fits ~10GB
else:
    pipe.to("cuda")

gpu_lock = asyncio.Lock()  # serialize generations: one at a time on small GPUs

app = FastAPI(title="z-image-turbo-shim")
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")


# ---- OpenAI-compatible request schema ----
class ImageRequest(BaseModel):
    prompt: str
    model: str = "z-image-turbo"
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"  # WxH, or "auto"
    response_format: str = "b64_json"  # "b64_json" | "url"
    # extensions / accepted-and-ignored for client compatibility
    seed: int | None = None
    quality: str | None = None
    style: str | None = None
    user: str | None = None


def _err(message: str, type_: str = "invalid_request_error", status: int = 400):
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": type_}})


def parse_size(s: str) -> tuple[int, int]:
    if not s or s == "auto":
        return 1024, 1024
    w, h = s.lower().split("x")
    return int(w), int(h)


def _cleanup_old_images():
    cutoff = time.time() - IMG_TTL_SECONDS
    for name in os.listdir(IMG_DIR):
        path = os.path.join(IMG_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


@app.post("/v1/images/generations")
async def generate(req: ImageRequest, authorization: str | None = Header(default=None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        return _err("Incorrect API key provided.", "invalid_api_key", 401)

    try:
        width, height = parse_size(req.size)
    except Exception:
        return _err(f"Invalid size: {req.size!r}. Use 'WxH' (e.g. '1024x1024') or 'auto'.")

    if req.response_format not in ("b64_json", "url"):
        return _err(f"Invalid response_format: {req.response_format!r}.")

    data = []
    async with gpu_lock:  # protect the GPU from concurrent requests
        for i in range(req.n):
            generator = None
            if req.seed is not None:
                generator = torch.Generator("cuda").manual_seed(req.seed + i)

            def _run():
                return pipe(
                    prompt=req.prompt,
                    num_inference_steps=9,  # Turbo: ~8 DiT forwards
                    guidance_scale=0.0,  # MUST be 0 for the Turbo model
                    width=width,
                    height=height,
                    generator=generator,
                ).images[0]

            image = await asyncio.to_thread(_run)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png = buf.getvalue()

            if req.response_format == "url":
                name = f"{uuid.uuid4().hex}.png"
                with open(os.path.join(IMG_DIR, name), "wb") as f:
                    f.write(png)
                data.append({"url": f"{PUBLIC_BASE}/images/{name}"})
            else:
                data.append({"b64_json": base64.b64encode(png).decode()})

    if req.response_format == "url":
        _cleanup_old_images()

    return {"created": int(time.time()), "data": data}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
