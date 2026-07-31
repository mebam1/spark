#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meshy Retexture: image + model -> textured model (+PBR maps)
Docs: https://docs.meshy.ai/api/retexture

Usage examples:
  python meshy_retexture.py \
    --image /path/to/style.jpg \
    --model /path/to/model.glb \
    --outdir ./meshy_output \
    --pbr \
    --keep-uv

Notes:
- For large files, hosting on a public URL is preferred; this script also supports encoding local files as data URIs for convenience.
- UV preservation (keep-uv) is enabled by default; if the model has no UV, quality may suffer — try disabling it to let Meshy rebuild UV.
"""

import os
import sys
import time
import json
import base64
import mimetypes
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========= Configure here =========

# The key is read from the .env file at the repo root, or from the MESHY_API_KEY env var.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

API_KEY = os.getenv("MESHY_API_KEY", "")
# =================================

API_BASE = "https://api.meshy.ai/openapi/v1"
CREATE_URL = f"{API_BASE}/retexture"
GET_URL_TPL = f"{API_BASE}/retexture/{{task_id}}"

def assert_api_key():
    if not API_KEY or API_KEY == "PUT_YOUR_API_KEY_HERE":
        print("[ERROR] Please set MESHY_API_KEY in the .env file at the repo root or as an environment variable")
        sys.exit(1)

def guess_mime(path: Path, fallback: str) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or fallback

def to_data_uri(path: Path, mime: str) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

def make_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

def create_session() -> requests.Session:
    """Create a requests session with retry logic and better connection handling"""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

def create_retexture_task(
    model_uri: str,
    image_uri: str,
    enable_pbr: bool,
    keep_uv: bool,
    ai_model: str = "latest",
    session: Optional[requests.Session] = None
) -> str:
    if session is None:
        session = create_session()

    payload = {
        "model_url": model_uri,
        "image_style_url": image_uri,
        "enable_pbr": enable_pbr,
        "enable_original_uv": keep_uv,
        "ai_model": ai_model,  # latest / meshy-4 / meshy-5
    }

    # Print payload size for debugging
    payload_str = json.dumps(payload)
    payload_size_mb = len(payload_str) / (1024 * 1024)
    print(f"[debug] Payload size: {payload_size_mb:.2f} MB")

    try:
        resp = session.post(CREATE_URL, headers=make_headers(), data=payload_str, timeout=180)
        print(f"[debug] POST {CREATE_URL} -> HTTP {resp.status_code}")
        # Accept both 200 (OK) and 202 (Accepted) as success
        if resp.status_code not in (200, 202):
            print(f"[debug] Response body: {resp.text}")
            raise RuntimeError(f"Create task failed ({resp.status_code}): {resp.text}")
        task_id = resp.json().get("result")
        if not task_id:
            print(f"[debug] Response body: {resp.text}")
            raise RuntimeError(f"Create task got no 'result' id: {resp.text}")
        return task_id
    except requests.exceptions.SSLError as e:
        print(f"[ERROR] SSL Error occurred. This may be due to:")
        print("  1. Large file size causing connection timeout")
        print("  2. Network/firewall issues")
        print("  3. SSL certificate problems")
        print(f"[ERROR] Details: {e}")
        raise

def get_task(task_id: str) -> Dict[str, Any]:
    url = GET_URL_TPL.format(task_id=task_id)
    resp = requests.get(url, headers=make_headers(), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Get task failed ({resp.status_code}): {resp.text}")
    return resp.json()

def get_task_verbose(task_id: str) -> Dict[str, Any]:
    """Get task with verbose error output"""
    url = GET_URL_TPL.format(task_id=task_id)
    resp = requests.get(url, headers=make_headers(), timeout=60)
    print(f"[debug] GET {url} -> HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"[debug] Response body: {resp.text}")
        raise RuntimeError(f"Get task failed ({resp.status_code}): {resp.text}")
    return resp.json()

def poll_until_done(task_id: str, interval: float = 5.0) -> Dict[str, Any]:
    last_progress = -1
    while True:
        task = get_task(task_id)
        status = task.get("status")
        progress = task.get("progress")
        if progress is not None and progress != last_progress:
            print(f"[Task {task_id}] status={status}, progress={progress}%")
            last_progress = progress
        else:
            print(f"[Task {task_id}] status={status} ...")

        if status in ("SUCCEEDED", "FAILED", "CANCELED"):
            # Print full task response for debugging
            if status == "FAILED":
                print(f"[debug] Full task response:")
                print(json.dumps(task, indent=2))
            return task
        time.sleep(interval)

def safe_name(url: str) -> str:
    # Derive a safe filename from the URL
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    return name or "file.bin"

def download(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_name(url)
    out_path = out_dir / filename
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    print(f"[downloaded] {out_path}")
    return out_path

def main():
    assert_api_key()

    parser = argparse.ArgumentParser(description="Meshy Retexture: image + model -> textured model")
    parser.add_argument("--image", required=True, help="Style/reference image path (png/jpg/jpeg/webp)")
    parser.add_argument("--model", required=True, help="3D model path (.glb/.gltf/.obj/.fbx/.stl)")
    parser.add_argument("--outdir", default="./meshy_output", help="Output directory")
    parser.add_argument("--pbr", action="store_true", help="Enable PBR maps (metallic/roughness/normal)")
    parser.add_argument("--keep-uv", action="store_true", help="Preserve original UV mapping (only use if model has UV; off by default)")
    parser.add_argument("--ai-model", default="latest", choices=["latest", "meshy-4", "meshy-5"], help="AI model for retexture")
    args = parser.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()

    if not img_path.exists():
        print(f"[ERROR] image not found: {img_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERROR] model not found: {model_path}")
        sys.exit(1)

    # Check file sizes and warn if too large
    img_size_mb = img_path.stat().st_size / (1024 * 1024)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"[info] Image size: {img_size_mb:.2f} MB")
    print(f"[info] Model size: {model_size_mb:.2f} MB")

    if model_size_mb > 10:
        print(f"[WARN] Model file is large ({model_size_mb:.2f} MB). Encoded payload will be ~{model_size_mb * 1.37:.2f} MB.")
        print("[WARN] Large files may cause SSL/connection issues. Consider using a public URL instead.")

    # Build data URIs
    img_mime = guess_mime(img_path, "image/png")  # fallback to png
    model_mime = "model/gltf-binary"              # correct MIME type for GLB; lets Meshy detect format
    print("[info] Encoding files to base64 data URIs...")
    image_uri = to_data_uri(img_path, img_mime)
    model_uri = to_data_uri(model_path, model_mime)

    # Create session with retry logic
    session = create_session()

    print("[info] creating retexture task ...")
    task_id = create_retexture_task(
        model_uri=model_uri,
        image_uri=image_uri,
        enable_pbr=bool(args.pbr),
        keep_uv=args.keep_uv,
        ai_model=args.ai_model,
        session=session
    )
    print(f"[created] task_id = {task_id}")

    print("[info] polling task ...")
    task = poll_until_done(task_id, interval=5.0)

    status = task.get("status")
    if status != "SUCCEEDED":
        err = (task.get("task_error") or {}).get("message", "")
        print(f"[FAILED] status={status}, error={err}")
        sys.exit(2)

    # Download model & textures
    task_out_dir = out_dir / task_id
    model_urls = (task.get("model_urls") or {})
    texture_sets = task.get("texture_urls") or []

    if model_urls:
        for k, url in model_urls.items():
            try:
                print(f"[info] downloading model ({k}) ...")
                download(url, task_out_dir)
            except Exception as e:
                print(f"[warn] failed to download {k}: {e}")

    if texture_sets:
        tex_dir = task_out_dir / "textures"
        for i, tex in enumerate(texture_sets):
            for map_name, url in tex.items():
                try:
                    print(f"[info] downloading texture set {i} - {map_name} ...")
                    download(url, tex_dir)
                except Exception as e:
                    print(f"[warn] failed to download {map_name}: {e}")

    thumb = task.get("thumbnail_url")
    if thumb:
        try:
            print("[info] downloading preview thumbnail ...")
            download(thumb, task_out_dir)
        except Exception as e:
            print(f"[warn] failed to download thumbnail: {e}")

    print(f"[DONE] outputs saved to: {task_out_dir}")

if __name__ == "__main__":
    main()
