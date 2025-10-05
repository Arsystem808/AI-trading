# scripts/smoke_check.py
from core.strategy_router import analyze_asset

if __name__ == "__main__":
    res = analyze_asset("AAPL", "Краткосрочный", "Octopus")
    assert res["strategy"] == "Octopus"
    assert (
        "agents" in res and isinstance(res["agents"], list) and len(res["agents"]) >= 2
    )
    assert "unified_conf" in res
    print("[SMOKE] Octopus OK:", res["agents"], res["unified_conf"])
