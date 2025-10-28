# core/model_fetch.py — bundle-first safe loader

import os
import io
import tarfile
import logging
import pathlib
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse((u or "").strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _safe_extract_targz(bytes_ bytes, dest: pathlib.Path):
    dest.mkdir(parents=True, exist_ok=True)
    # Безопасная распаковка: защита от path traversal
    with tarfile.open(fileobj=io.BytesIO(bytes_data), mode="r:*") as tar:
        for m in tar.getmembers():
            target = dest / m.name
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise RuntimeError(f"Unsafe path in tar: {m.name}")
        tar.extractall(dest)
    log.info("✓ Bundle extracted to %s", dest)

def ensure_models():
    dest = pathlib.Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
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

    # Фолбэк: пофайловые ссылки, если заданы (можно удалить, если не нужно)
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
