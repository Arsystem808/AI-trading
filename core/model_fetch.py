from urllib.parse import urlparse
import logging, os, pathlib, requests

def _is_valid_url(u: str) -> bool:
    try:
        u = (u or "").strip()
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def ensure_models():
    dest = pathlib.Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)
    ASSETS = {
        "alphapulse_AAPL.joblib": os.getenv("MODEL_AAPL_URL"),
        "alphapulse_ETHUSD.joblib": os.getenv("MODEL_ETH_URL"),
    }
    for fname, url in ASSETS.items():
        if not url or not _is_valid_url(url):
            logging.warning("Пропускаю %s: некорректный или пустой URL в Secrets", fname)
            continue
        path = dest / fname
        if path.exists() and path.stat().st_size > 0:
            logging.info("✓ Модель %s уже существует", fname)
            continue
        try:
            r = requests.get(url.strip(), timeout=180)
            r.raise_for_status()
            path.write_bytes(r.content)
            logging.info("✓ Модель %s загружена (%d байт)", fname, len(r.content))
        except Exception as e:
            logging.error("✗ Ошибка загрузки %s: %s", fname, e)
            # не роняем приложение
            continue
