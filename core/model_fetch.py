# core/model_fetch.py — minimal safe loader (no type hints)

import os
import io
import tarfile
import logging
from pathlib import Path
from urllib.parse import urlparse
import requests

log = logging.getLogger(__name__)

def _is_valid_url(u):
    try:
        p = urlparse((u or "").strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _safe_extract_targz(bytes_data, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bytes_data), mode="r:*") as tar:
        base = dest.resolve()
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError("Unsafe path in tar: %s" % m.name)
        tar.extractall(dest)
    log.info("✓ Bundle extracted to %s", dest)

def ensure_models():
    dest = Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)

    bundle_url = (os.getenv("MODEL_BUNDLE_URL") or "").strip()
    if bundle_url and _is_valid_url(bundle_url):
        try:
            log.info("⬇ Downloading model bundle from %s", bundle_url)
            r = requests.get(bundle_url, timeout=240, allow_redirects=True)
            r.raise_for_status()
            _safe_extract_targz(r.content, dest)
            return
        except Exception as e:
            log.warning("Bundle download failed: %s", e)
    elif bundle_url:
        log.warning("Invalid bundle URL in env: %s", bundle_url)

    assets = {
        "alphapulse_AAPL.joblib": os.getenv("MODEL_AAPL_URL"),
        "alphapulse_ETHUSD.joblib": os.getenv("MODEL_ETH_URL"),
    }
    for fname, url in assets.items():
        if not url or not _is_valid_url(url):
            log.warning("Пропускаю %s: некорректный или пустой URL в Secrets", fname)
            continue
        path = dest / fname
        if path.exists() and path.stat().st_size > 0:
            log.info("✓ %s уже существует", fname)
            continue
        try:
            log.info("⬇ Downloading %s", fname)
            rr = requests.get(url.strip(), timeout=180, allow_redirects=True)
            rr.raise_for_status()
            path.write_bytes(rr.content)
            log.info("✓ %s saved (%d bytes)", fname, len(rr.content))
        except Exception as e:
            log.error("✗ Ошибка загрузки %s: %s", fname, e)

