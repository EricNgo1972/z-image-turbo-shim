# OpenAI-compatible image-generation shim for Tongyi-MAI/Z-Image-Turbo.
#
# Exposes POST /v1/images/generations matching the OpenAI Images API, so any
# OpenAI/Azure SDK works by swapping the base URL and model name.
import asyncio
import base64
import contextlib
import io
import os
import time
import uuid

import torch
from diffusers import ZImagePipeline
from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

# ---- config (via env) ----
MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
# Name advertised via /v1/models and the one clients should send as "model".
MODEL_NAME = os.getenv("MODEL_NAME", "z-image-turbo")
API_KEY = os.getenv("API_KEY")  # if set, require "Authorization: Bearer <API_KEY>"
PUBLIC_BASE = os.getenv("PUBLIC_BASE", "http://localhost:8000")  # for response_format=url
DTYPE = os.getenv("DTYPE", "bfloat16")  # compute dtype: "bfloat16" | "float16"
# "none" | "fp8". fp8 = torchao float8 weight-only quant (~6GB, fully on-GPU on 10GB cards).
QUANTIZATION = os.getenv("QUANTIZATION", "none").lower()
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "1") == "1"  # keep VRAM low (ignored when fp8)
IMG_DIR = os.getenv("IMG_DIR", "images")
IMG_TTL_SECONDS = int(os.getenv("IMG_TTL_SECONDS", "3600"))  # url-mode cleanup age
# Max requests allowed in the GPU queue (1 running + the rest waiting). Beyond this,
# new requests get HTTP 429 instead of piling up. Keep small on single-GPU boxes.
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "8"))
RETRY_AFTER = int(os.getenv("RETRY_AFTER", "10"))  # seconds hint sent with 429

os.makedirs(IMG_DIR, exist_ok=True)


def _quantize_fp8(pipeline):
    """Apply torchao float8 weight-only quantization to the heavy submodels.

    Loads in `DTYPE`, quantizes weights to float8 in place (~halves VRAM vs bf16),
    so the 6B transformer + text encoder fit fully on a 10GB GPU. Quantizing before
    .to("cuda") keeps the move small.
    """
    from torchao.quantization import float8_weight_only, quantize_

    for attr in ("transformer", "text_encoder", "text_encoder_2"):
        comp = getattr(pipeline, attr, None)
        if comp is not None:
            quantize_(comp, float8_weight_only())
            print(f"[startup] quantized {attr} -> float8 weight-only")


# ---- load model ONCE at startup ----
_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=_dtype, low_cpu_mem_usage=False)

if QUANTIZATION == "fp8":
    _quantize_fp8(pipe)
    pipe.to("cuda")  # ~6GB, fits 10GB fully on-GPU (no offload needed -> faster)
    print("[startup] fp8 mode: model resident on GPU")
elif CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()  # streams weights to GPU as needed -> fits ~10GB
else:
    pipe.to("cuda")

# img2img pipeline for /v1/images/edits. Reuses the SAME (already-quantized) model
# components via from_pipe -> no extra VRAM. Degrades gracefully on older diffusers.
try:
    from diffusers import ZImageImg2ImgPipeline

    img2img_pipe = ZImageImg2ImgPipeline.from_pipe(pipe)
    EDITS_AVAILABLE = True
    print("[startup] img2img (/v1/images/edits) enabled")
except Exception as e:  # pragma: no cover - depends on installed diffusers version
    img2img_pipe = None
    EDITS_AVAILABLE = False
    print(f"[startup] img2img unavailable, /v1/images/edits disabled: {e}")

class QueueFull(Exception):
    """Raised when the GPU queue is at capacity."""


class GpuQueue:
    """Serializes GPU work (one job at a time) with a bounded waiting queue.

    `depth` counts everything in the system (the running job + waiters). Acquiring
    a slot beyond `limit` raises QueueFull so the caller can return HTTP 429 instead
    of letting requests pile up unbounded. Single-event-loop only (run --workers 1).
    """

    def __init__(self, limit: int):
        self._lock = asyncio.Lock()
        self._limit = limit
        self._depth = 0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def running(self) -> bool:
        return self._lock.locked()

    @contextlib.asynccontextmanager
    async def slot(self):
        # No await between the check and increment -> atomic on one event loop.
        if self._depth >= self._limit:
            raise QueueFull()
        self._depth += 1
        try:
            async with self._lock:
                yield
        finally:
            self._depth -= 1


gpu_queue = GpuQueue(MAX_QUEUE)


def _busy():
    return JSONResponse(
        status_code=429,
        content={"error": {
            "message": "Server busy: GPU queue is full. Retry shortly.",
            "type": "rate_limit_exceeded",
        }},
        headers={"Retry-After": str(RETRY_AFTER)},
    )

app = FastAPI(title="z-image-turbo-shim")
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def ui():
    """Minimal browser UI for testing image generation."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


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


def _encode_image(image, response_format: str) -> dict:
    """Serialize a PIL image to an OpenAI-style data entry (b64_json or url)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png = buf.getvalue()
    if response_format == "url":
        name = f"{uuid.uuid4().hex}.png"
        with open(os.path.join(IMG_DIR, name), "wb") as f:
            f.write(png)
        return {"url": f"{PUBLIC_BASE}/images/{name}"}
    return {"b64_json": base64.b64encode(png).decode()}


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
    try:
        async with gpu_queue.slot():  # serialize + bounded queue (429 when full)
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
                data.append(_encode_image(image, req.response_format))
    except QueueFull:
        return _busy()

    if req.response_format == "url":
        _cleanup_old_images()

    return {"created": int(time.time()), "data": data}


def _build_edit_mask(mask_bytes: bytes, size: tuple[int, int]):
    """OpenAI mask semantics: areas to EDIT are transparent (alpha 0).

    Returns an L-mode mask where 255 = edit (use generated), 0 = keep (use original),
    suitable for Image.composite(generated, original, mask).
    """
    m = Image.open(io.BytesIO(mask_bytes)).resize(size)
    if "A" in m.getbands():
        src = m.getchannel("A")  # transparent -> edit
    else:
        src = m.convert("L")  # fallback convention: dark -> edit
    return src.point(lambda v: 255 if v < 128 else 0)


@app.post("/v1/images/edits")
async def edits(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: UploadFile | None = File(default=None),
    model: str = Form("z-image-turbo"),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"),
    strength: float = Form(0.6),  # extension: img2img denoise strength (0..1)
    seed: int | None = Form(default=None),  # extension
    authorization: str | None = Header(default=None),
):
    """OpenAI-compatible image edit (img2img). Optional mask = best-effort inpaint.

    Multipart/form-data, mirroring POST /v1/images/edits.
    """
    if API_KEY and authorization != f"Bearer {API_KEY}":
        return _err("Incorrect API key provided.", "invalid_api_key", 401)
    if not EDITS_AVAILABLE:
        return _err("Image edits not supported by this server's diffusers version.",
                    "not_supported", 501)
    if not 1 <= n <= 4:
        return _err("n must be between 1 and 4.")
    if response_format not in ("b64_json", "url"):
        return _err(f"Invalid response_format: {response_format!r}.")
    strength = max(0.0, min(1.0, strength))

    try:
        width, height = parse_size(size)
    except Exception:
        return _err(f"Invalid size: {size!r}. Use 'WxH' (e.g. '1024x1024') or 'auto'.")

    # Soft early reject so we don't buffer the upload when the queue is already full.
    if gpu_queue.depth >= MAX_QUEUE:
        return _busy()

    init = Image.open(io.BytesIO(await image.read())).convert("RGB").resize((width, height))
    edit_mask = None
    if mask is not None:
        edit_mask = _build_edit_mask(await mask.read(), (width, height))

    data = []
    try:
        async with gpu_queue.slot():  # serialize + bounded queue (429 when full)
            for i in range(n):
                generator = None
                if seed is not None:
                    generator = torch.Generator("cuda").manual_seed(seed + i)

                def _run():
                    return img2img_pipe(
                        prompt=prompt,
                        image=init,
                        strength=strength,
                        num_inference_steps=9,
                        guidance_scale=0.0,
                        generator=generator,
                    ).images[0]

                result = await asyncio.to_thread(_run)
                result = result.resize((width, height))
                if edit_mask is not None:
                    # keep original outside the mask (best-effort inpaint; Turbo isn't
                    # inpaint-trained, so this composites rather than context-fills)
                    result = Image.composite(result, init, edit_mask)
                data.append(_encode_image(result, response_format))
    except QueueFull:
        return _busy()

    if response_format == "url":
        _cleanup_old_images()

    return {"created": int(time.time()), "data": data}


def _model_object():
    # OpenAI "model" object shape. created=0 (stable; no wall-clock at import).
    return {
        "id": MODEL_NAME,
        "object": "model",
        "created": 0,
        "owned_by": "z-image-turbo-shim",
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [_model_object()]}


@app.get("/v1/models/{model}")
def retrieve_model(model: str):
    if model != MODEL_NAME:
        return _err(f"The model '{model}' does not exist.", "invalid_request_error", 404)
    return _model_object()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "served_as": MODEL_NAME,
        "dtype": DTYPE,
        "quantization": QUANTIZATION,
        "edits": EDITS_AVAILABLE,
        "queue": {
            "running": gpu_queue.running,
            "depth": gpu_queue.depth,  # running + waiting
            "limit": MAX_QUEUE,
        },
    }
