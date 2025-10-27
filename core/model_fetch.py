import os, pathlib, requests
DEST = pathlib.Path(os.getenv("ARXORA_MODEL_DIR", "/tmp/models"))
ASSETS = {
    "alphapulse_AAPL.joblib": os.getenv("MODEL_AAPL_URL"),
    "alphapulse_ETHUSD.joblib": os.getenv("MODEL_ETH_URL"),
}
def ensure_models():
    DEST.mkdir(parents=True, exist_ok=True)
    for fname, url in ASSETS.items():
        if not url: continue
        path = DEST / fname
        if path.exists(): continue
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
