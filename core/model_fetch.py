import logging
logging.basicConfig(level=logging.INFO)

def ensure_models():
    DEST.mkdir(parents=True, exist_ok=True)
    for fname, url in ASSETS.items():
        if not url:
            continue
        path = DEST / fname
        if path.exists() and path.stat().st_size > 0:
            continue
        logging.info(f"Downloading {fname} -> {path}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)
