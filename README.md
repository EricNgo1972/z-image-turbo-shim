# z-image-turbo-shim

An OpenAI-compatible image-generation API in front of the self-hosted
[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) model.

It exposes **`POST /v1/images/generations`** matching the
[OpenAI Images API](https://platform.openai.com/docs/api-reference/images), so existing
OpenAI / Azure OpenAI SDK code works by swapping the **base URL** and **model name** —
no client rewrite needed.

```
[your app] --HTTP (OpenAI Images API)--> [z-image-turbo-shim] --> [GPU / Z-Image-Turbo]
```

Z-Image-Turbo is released under **Apache 2.0**, so generated images are fine for
commercial use (e.g. social media posting).

## Requirements

- NVIDIA GPU. Z-Image-Turbo fits ~10 GB VRAM with `CPU_OFFLOAD=1` (default).
  Use the **FP8** checkpoint / `DTYPE=float16` if you hit out-of-memory.
- Python 3.10+ (or Docker + NVIDIA Container Toolkit).

## Run (local)

```bash
pip install -r requirements.txt

# optional config
export API_KEY=sk-local-changeme
export PUBLIC_BASE=http://your-gpu-server:8000

uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

First start downloads the model (~minutes); subsequent requests are fast (Turbo = 8 steps).
Keep `--workers 1` so the model loads once and small GPUs don't OOM.

## Run (Docker)

```bash
docker build -t z-image-turbo-shim .
docker run --gpus all -p 8000:8000 \
  -e API_KEY=sk-local-changeme \
  -e PUBLIC_BASE=http://your-gpu-server:8000 \
  z-image-turbo-shim
```

## Deploy to your GPU server

Assumes the server has an NVIDIA GPU + driver. Replace `you@your-gpu-server` and paths
to match your box.

### Step 1 — Get the code onto the server

```bash
# on the GPU server
git clone https://github.com/EricNgo1972/z-image-turbo-shim.git
cd z-image-turbo-shim
cp .env.example .env        # then edit .env: set API_KEY, PUBLIC_BASE, etc.
```

(For private changes you can also `scp -r ./z-image-turbo-shim you@your-gpu-server:~/`.)

### Option A — Docker (recommended)

Requires Docker + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(`nvidia-ctk`) installed on the host. Verify GPU access first:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Build and run (model weights cached in a named volume so they survive restarts):

```bash
docker build -t z-image-turbo-shim .

docker run -d --name z-image-shim --restart unless-stopped \
  --gpus all -p 8000:8000 \
  --env-file .env \
  -v hf-cache:/root/.cache/huggingface \
  z-image-turbo-shim

docker logs -f z-image-shim        # watch first-run model download
```

Update later:

```bash
git pull && docker build -t z-image-turbo-shim . \
  && docker rm -f z-image-shim \
  && docker run -d --name z-image-shim --restart unless-stopped \
       --gpus all -p 8000:8000 --env-file .env \
       -v hf-cache:/root/.cache/huggingface z-image-turbo-shim
```

### Option B — Bare metal + systemd

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Create `/etc/systemd/system/z-image-shim.service` (adjust `User` and `WorkingDirectory`):

```ini
[Unit]
Description=z-image-turbo-shim
After=network-online.target
Wants=network-online.target

[Service]
User=you
WorkingDirectory=/home/you/z-image-turbo-shim
EnvironmentFile=/home/you/z-image-turbo-shim/.env
ExecStart=/home/you/z-image-turbo-shim/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now z-image-shim
journalctl -u z-image-shim -f        # watch logs / first-run download
```

### Step 2 — Verify the deploy

From the server (or any machine that can reach it):

```bash
BASE_URL=http://localhost:8000 API_KEY=sk-local-changeme python test_client.py
```

### Step 3 — Make it reachable + safe

- **Open the port / firewall:** allow TCP 8000 only from trusted sources
  (e.g. `sudo ufw allow from <app-server-ip> to any port 8000`).
- **Always set `API_KEY`** if the port is reachable beyond localhost.
- **TLS (recommended):** put nginx/Caddy in front to terminate HTTPS and proxy to
  `127.0.0.1:8000`, then point `PUBLIC_BASE` and your .NET `Endpoint` at the HTTPS URL.

## Configuration

See [`.env.example`](.env.example). Key vars:

| Var | Default | Purpose |
|---|---|---|
| `API_KEY` | _(unset)_ | If set, require `Authorization: Bearer <API_KEY>`. |
| `PUBLIC_BASE` | `http://localhost:8000` | URL clients use to fetch images when `response_format=url`. |
| `MODEL_ID` | `Tongyi-MAI/Z-Image-Turbo` | Model to load. |
| `MODEL_NAME` | `z-image-turbo` | Name advertised via `/v1/models` and expected as the request `model`. |
| `DTYPE` | `bfloat16` | `bfloat16` or `float16`. |
| `CPU_OFFLOAD` | `1` | Stream weights to GPU on demand (low VRAM). |
| `IMG_DIR` | `images` | Where `url`-mode images are written. |
| `IMG_TTL_SECONDS` | `3600` | Auto-delete `url`-mode images older than this. |

## API

### `POST /v1/images/generations`

Request (OpenAI-compatible; extra fields accepted and ignored):

```json
{
  "prompt": "a cozy coffee shop at golden hour, photorealistic, 85mm",
  "model": "z-image-turbo",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json",
  "seed": 42
}
```

Response:

```json
{ "created": 1750000000, "data": [ { "b64_json": "iVBORw0KGgo..." } ] }
```

`size` accepts `WxH` (e.g. `1024x1024`, `1024x1536`, `1536x1024`) or `auto`.
`response_format` is `b64_json` (default) or `url`.

### `GET /v1/models`

Advertises the single served model (OpenAI-compatible discovery).

```json
{
  "object": "list",
  "data": [
    { "id": "z-image-turbo", "object": "model", "created": 0, "owned_by": "z-image-turbo-shim" }
  ]
}
```

`GET /v1/models/{model}` returns the same object for `z-image-turbo`, or 404 otherwise.

### `GET /health`

```json
{ "status": "ok", "model": "Tongyi-MAI/Z-Image-Turbo", "served_as": "z-image-turbo" }
```

## Client examples

### .NET (official OpenAI SDK)

```csharp
// dotnet add package OpenAI
using OpenAI.Images;
using System.ClientModel;

var client = new ImageClient(
    model: "z-image-turbo",
    credential: new ApiKeyCredential("sk-local-changeme"),
    options: new OpenAI.OpenAIClientOptions {
        Endpoint = new Uri("http://your-gpu-server:8000/v1")
    });

var result = await client.GenerateImageAsync(
    "a cozy coffee shop at golden hour, photorealistic, 85mm",
    new ImageGenerationOptions {
        Size = GeneratedImageSize.W1024xH1024,
        ResponseFormat = GeneratedImageFormat.Bytes
    });

await File.WriteAllBytesAsync("out.png", result.Value.ImageBytes.ToArray());
```

### curl

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer sk-local-changeme" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red panda astronaut, studio lighting","size":"1024x1024"}' \
  | python -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

## Smoke test

After the server is up, verify the whole path (health → models → one image):

```bash
python test_client.py
# or against a remote box with auth:
BASE_URL=http://your-gpu-server:8000 API_KEY=sk-local-changeme python test_client.py
```

Stdlib-only (no pip install). Saves `smoke_test.png` and exits non-zero on any failure,
so it works in CI / deploy checks.

## Notes & limits

- **One generation at a time.** A GPU lock serializes requests so small cards don't OOM;
  `n > 1` runs sequentially. Set generous client timeouts.
- **Turbo settings are fixed** (`guidance_scale=0.0`, 9 steps) — correct for this model.
- Tune your prompt for quality: add lens, lighting, and material detail.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The Z-Image-Turbo model weights are Apache 2.0 by
Tongyi-MAI — review their model card for terms.
