cat > scripts/smoke_check.py <<'PY'
import core.strategy as s
for m in ("Global","W7","M7"):
    r = s.analyze_asset("SPY","Краткосрочный", m)
    print(m, r["confidence_breakdown"], r["levels"])
PY
