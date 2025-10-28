import os, pathlib, requests, logging
logging.basicConfig(level=logging.INFO)

ASSETS = {
    "alphapulse_AAPL.joblib": os.getenv("MODEL_AAPL_URL"),
    "alphapulse_ETHUSD.joblib": os.getenv("MODEL_ETH_URL"),
}

def ensure_models():
    dest = pathlib.Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
    dest.mkdir(parents=True, exist_ok=True)
    for fname, url in ASSETS.items():
        if not url:
            continue
        path = dest / fname
        if path.exists() and path.stat().st_size > 0:
            continue
        logging.info(f"Downloading {fname} -> {path}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)
