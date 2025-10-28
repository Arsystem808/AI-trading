# core/model_fetch.py — bundle-first loader with filename+SHA marker

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

def _read_marker(marker: Path):
    name, sh = None, None
    if marker.exists():
        try:
            for line in marker.read_text(errors="ignore").splitlines():
                if line.startswith("FILENAME="):
                    name = line.split("=", 1)[1].strip()
                elif line.startswith("SHA256="):
                    sh = line.split("=", 1)[1].strip()
        except Exception:
            pass
    return name, sh

def _write_marker(marker: Path, name: str, sh: str):
    try:
        marker.write_text(f"FILENAME={name}\nSHA256={sh}\n")
    except Exception as e:
        log.warning("Write marker failed: %s", e)

# ---------- Public API ----------
def ensure_models():
    """
    Скачивает и распаковывает бандл моделей в ARXORA_MODEL_DIR (по умолчанию /tmp/models).
    - Пропускает загрузку, если .bundle_version совпадает по имени файла и SHA256.
    - MODEL_BUNDLE_SHA256 (опционально) усиливает проверку; при отсутствии вычисляется из скачанного архива.
    - Фолбэк: пофайловые ссылки MODEL_AAPL_URL/MODEL_ETH_URL.
    """
    dest = Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)

    bundle_url = (os.getenv("MODEL_BUNDLE_URL") or "").strip()
    conf_sha = (os.getenv("MODEL_BUNDLE_SHA256") or "").strip()
    marker = dest / ".bundle_version"

    if bundle_url and _is_valid_url(bundle_url):
        ver = os.path.basename(bundle_url.split("?", 1)[0])

        # Пропуск, если совпадает имя и (если задан) SHA, и каталог не пустой
        m_name, m_sha = _read_marker(marker)
        try:
            any_item = next(dest.iterdir(), None)
        except Exception:
            any_item = None
        if any_item and m_name == ver and (not conf_sha or (m_sha and m_sha.lower() == conf_sha.lower())):
            log.info("Bundle already present (%s) — skipping download", ver)
            return

        try:
            log.info("⬇ Downloading model bundle from %s", bundle_url)
            r = requests.get(bundle_url, timeout=240, allow_redirects=True)
            r.raise_for_status()
            data = r.content

            calc_sha = sha256(data).hexdigest()
            if conf_sha and calc_sha.lower() != conf_sha.lower():
                log.warning("Bundle SHA mismatch: expected %s got %s — skipping extract", conf_sha, calc_sha)
                return

            _safe_extract_targz(data, dest)
            _write_marker(marker, ver, calc_sha)
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
