# core/model_fetch.py — bundle-first loader with version marker

import os
import io
import tarfile
import logging
from pathlib import Path
from urllib.parse import urlparse
import requests
from hashlib import sha256

log = logging.getLogger(__name__)

# ---------- Utils ----------
def _is_valid_url(u):
    try:
        p = urlparse((u or "").strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _safe_extract_targz(bytes_data, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bytes_data), mode="r:*") as tar:
        base = dest.resolve()
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError("Unsafe path in tar: %s" % m.name)
        tar.extractall(dest)
    log.info("✓ Bundle extracted to %s", dest)

def _verify_sha256( bytes, expected_hex: str) -> bool:
    try:
        if not expected_hex:
            return True
        h = sha256(data).hexdigest()
        if h.lower() != expected_hex.lower():
            log.warning("Bundle SHA256 mismatch: expected %s, got %s", expected_hex, h)
            return False
        return True
    except Exception as e:
        log.warning("SHA256 check failed: %s", e)
        return False

# ---------- Public API ----------
def ensure_models():
    """
    Скачивает и распаковывает бандл моделей в ARXORA_MODEL_DIR (по умолчанию /tmp/models).
    - Если версия бандла уже отмечена в .bundle_version — загрузка пропускается.
    - Опционально проверяет MODEL_BUNDLE_SHA256.
    - Фолбэк: пофайловые ссылки MODEL_AAPL_URL/MODEL_ETH_URL (если заданы).
    """
    dest = Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)

    bundle_url = (os.getenv("MODEL_BUNDLE_URL") or "").strip()
    bundle_sha = (os.getenv("MODEL_BUNDLE_SHA256") or "").strip()
    marker = dest / ".bundle_version"

    if bundle_url and _is_valid_url(bundle_url):
        ver = os.path.basename(bundle_url)  # например: models-20251028_030448.tar.gz
        # Если версия совпала и каталог не пустой — пропускаем
        if marker.exists() and marker.read_text(errors="ignore").strip() == ver:
            try:
                any_item = next(dest.iterdir(), None)
            except Exception:
                any_item = None
            if any_item:
                log.info("Bundle already present (%s) — skipping download", ver)
                return

        try:
            log.info("⬇ Downloading model bundle from %s", bundle_url)
            r = requests.get(bundle_url, timeout=240, allow_redirects=True)
            r.raise_for_status()
            data = r.content

            # Проверка целостности (опционально)
            if bundle_sha and not _verify_sha256(data, bundle_sha):
                log.warning("Bundle integrity failed — skipping extract")
            else:
                _safe_extract_targz(data, dest)
                marker.write_text(ver)
                log.info("Bundle version marked: %s", ver)
                return
        except Exception as e:
            log.warning("Bundle download failed: %s", e)
    elif bundle_url:
        log.warning("Invalid bundle URL in env: %s", bundle_url)

    # ---- Fallback: per-file assets (опционально) ----
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

