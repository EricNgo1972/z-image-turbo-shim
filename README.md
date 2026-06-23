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

## Notes & limits

- **One generation at a time.** A GPU lock serializes requests so small cards don't OOM;
  `n > 1` runs sequentially. Set generous client timeouts.
- **Turbo settings are fixed** (`guidance_scale=0.0`, 9 steps) — correct for this model.
- Tune your prompt for quality: add lens, lighting, and material detail.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The Z-Image-Turbo model weights are Apache 2.0 by
Tongyi-MAI — review their model card for terms.
