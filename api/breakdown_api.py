import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/signal")
def signal(ticker: str = Query(..., min_length=1)):
    # Запускаем ваш скрипт и возвращаем его JSON
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    proc = subprocess.run(
        [sys.executable, "scripts/signal_with_confidence.py", ticker], env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip(), "ticker": ticker}
    return json.loads(proc.stdout)


from fastapi.middleware.cors import CORSMiddleware

# Разрешить фронтенду доступ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # в проде сузьте домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
