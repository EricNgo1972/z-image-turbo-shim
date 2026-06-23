#!/usr/bin/env python3
"""Smoke-test for a running z-image-turbo-shim instance.

Checks /health and /v1/models, then generates one image via
/v1/images/generations and saves it. Exits non-zero on any failure.

Usage:
    python test_client.py
    BASE_URL=http://your-gpu-server:8000 API_KEY=sk-local-changeme python test_client.py

Requires only the stdlib (urllib) so it can run anywhere, no pip install.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY")  # set if the server enforces auth
MODEL = os.getenv("MODEL_NAME", "z-image-turbo")
OUT = os.getenv("OUT", "smoke_test.png")
PROMPT = os.getenv("PROMPT", "a red panda astronaut, studio lighting, photorealistic")
TIMEOUT = float(os.getenv("TIMEOUT", "180"))  # generation can be slow on small GPUs


def _request(method, path, body=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.status, json.loads(resp.read().decode())


def _ok(msg):
    print(f"  \033[32mPASS\033[0m {msg}")


def _fail(msg):
    print(f"  \033[31mFAIL\033[0m {msg}")
    sys.exit(1)


def main():
    print(f"Target: {BASE_URL}  (model={MODEL})\n")

    # 1) health
    print("[1/3] GET /health")
    try:
        status, body = _request("GET", "/health")
        assert status == 200 and body.get("status") == "ok", body
        _ok(f"healthy, model={body.get('model')}")
    except Exception as e:
        _fail(f"/health: {e}")

    # 2) models advertisement
    print("[2/3] GET /v1/models")
    try:
        status, body = _request("GET", "/v1/models")
        ids = [m.get("id") for m in body.get("data", [])]
        assert status == 200 and MODEL in ids, body
        _ok(f"advertises {ids}")
    except Exception as e:
        _fail(f"/v1/models: {e}")

    # 3) generate one image
    print("[3/3] POST /v1/images/generations")
    try:
        status, body = _request(
            "POST",
            "/v1/images/generations",
            {
                "model": MODEL,
                "prompt": PROMPT,
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json",
                "seed": 42,
            },
        )
        b64 = body["data"][0]["b64_json"]
        png = base64.b64decode(b64)
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a valid PNG"
        with open(OUT, "wb") as f:
            f.write(png)
        _ok(f"generated {len(png):,} bytes -> {OUT}")
    except urllib.error.HTTPError as e:
        _fail(f"generation HTTP {e.code}: {e.read().decode(errors='replace')}")
    except Exception as e:
        _fail(f"generation: {e}")

    print("\n\033[32mAll smoke tests passed.\033[0m")


if __name__ == "__main__":
    main()
