# core/model_fetch.py — bundle-first loader
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
        tar.extractall(dest)  # распаковка бандла моделей
    logging.info("✓ Bundle extracted to %s", dest)

def ensure_models():
    dest = pathlib.Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)

    bundle_url = os.getenv("MODEL_BUNDLE_URL", "").strip()
    if bundle_url and _is_valid_url(bundle_url):
        try:
            logging.info("⬇ Downloading model bundle from %s", bundle_url)
            r = requests.get(bundle_url, timeout=240, allow_redirects=True)
            r.raise_for_status()
            _extract_targz(r.content, dest)
            return
        except Exception as e:
            logging.warning("Bundle download failed: %s", e)

    # Fallback: пофайловые ссылки, если заданы
    assets = {
        "alphapulse_AAPL.joblib": os.getenv("MODEL_AAPL_URL"),
        "alphapulse_ETHUSD.joblib": os.getenv("MODEL_ETH_URL"),
    }
    for fname, url in assets.items():
        if not url or not _is_valid_url(url):
            logging.warning("Пропускаю %s: некорректный или пустой URL в Secrets", fname)
            continue
        path = dest / fname
        if path.exists() and path.stat().st_size > 0:
            logging.info("✓ %s уже существует", fname)
            continue
        try:
            logging.info("⬇ Downloading %s", fname)
            rr = requests.get(url.strip(), timeout=180, allow_redirects=True)
            rr.raise_for_status()
            path.write_bytes(rr.content)
            logging.info("✓ %s saved (%d bytes)", fname, len(rr.content))
        except Exception as e:
            logging.error("✗ Ошибка загрузки %s: %s", fname, e)
