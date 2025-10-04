from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

STORE = Path("snapshots_store")
INDEX = STORE / "index.csv"
STORE.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _hash_weights(weights: Dict[str, float] | None) -> str:
    if not weights:
        return "none"
    s = json.dumps(weights, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def save_snapshot(payload: Dict[str, Any]) -> str:
    meta = {
        "generated_at": _now_iso(),
        "ticker": payload.get("ticker"),
        "horizon": payload.get("horizon"),
        "model": payload.get("model"),
        "model_version": payload.get("model_version", "v1"),
        "weights_hash": _hash_weights(payload.get("weights")),
        "data_window": payload.get("data_window", {}),
    }
    body = {
        "recommendation": payload.get("recommendation"),
        "levels": payload.get("levels"),
        "context": payload.get("context", []),
        "agents": payload.get("agents", []),
        "weights": payload.get("weights", {}),
        "last_price": payload.get("last_price", 0.0),
        "note_html": payload.get("note_html", ""),
        "alt": payload.get("alt", ""),
    }
    raw: Dict[str, Any] = {"meta": meta, "body": body}
    sid = hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    raw["snapshot_id"] = sid

    (STORE / f"{sid}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    if not INDEX.exists():
        with open(INDEX, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["snapshot_id", "generated_at", "ticker", "horizon", "model", "model_version", "weights_hash"]
            )
    with open(INDEX, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                sid,
                meta["generated_at"],
                meta["ticker"],
                meta["horizon"],
                meta["model"],
                meta["model_version"],
                meta["weights_hash"],
            ]
        )
    return sid


def load_snapshot(sid: str) -> Dict[str, Any]:
    p = STORE / f"{sid}.json"
    if not p.exists():
        raise FileNotFoundError(f"snapshot {sid} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def compare_snapshots(a_id: str, b_id: str) -> Dict[str, Any]:
    a = load_snapshot(a_id)
    b = load_snapshot(b_id)

    def rec(x: Dict[str, Any]) -> Dict[str, Any]:
        return x["body"]["recommendation"]

    def lv(x: Dict[str, Any]) -> Dict[str, Any]:
        return x["body"].get("levels") or {}

    def safe_conf(x: Dict[str, Any]) -> float:
        return float(rec(x).get("confidence", 0.5))

    def safe_act(x: Dict[str, Any]) -> str:
        return str(rec(x).get("action", "WAIT"))

    delta = {
        "action": {"a": safe_act(a), "b": safe_act(b)},
        "confidence": {"a": safe_conf(a), "b": safe_conf(b), "diff": safe_conf(b) - safe_conf(a)},
        "levels": {"a": lv(a), "b": lv(b)},
        "meta": {"a": a["meta"], "b": b["meta"]},
    }
    return {"a": a, "b": b, "delta": delta}

