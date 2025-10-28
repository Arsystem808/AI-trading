# core/model_fetch.py
import os, pathlib, logging, tarfile, io
from urllib.parse import urlparse
import requests

logging.basicConfig(level=logging.INFO)

def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse((u or "").strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _extract_targz(bytes_ bytes, dest: pathlib.Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bytes_data), mode="r:gz") as tar:
        tar.extractall(dest)
    logging.info("✓ Bundle extracted to %s", dest)

def ensure_models():
    dest = pathlib.Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)
    bundle_url = (os.getenv("MODEL_BUNDLE_URL") or "").strip()
    if bundle_url and _is_valid_url(bundle_url):
        try:
            logging.info("⬇ Downloading model bundle from %s", bundle_url)
            r = requests.get(bundle_url, timeout=240, allow_redirects=True)
            r.raise_for_status()
            _extract_targz(r.content, dest)
            return
        except Exception as e:
            logging.warning("Bundle download failed: %s", e)
    logging.info("No valid bundle URL; skipping model download.")
