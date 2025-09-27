# -*- coding: utf-8 -*-
# app.py — ARXORA MVP: динамический импорт стратегий, снапшоты с valid_until, без выбора горизонта

import os, re, json, csv, hashlib, traceback, importlib
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Page / Branding =====
st.set_page_config(page_title="Arxora — трейд‑ИИ (MVP)", page_icon="assets/arxora_favicon_512.png", layout="centered")

def render_arxora_header():
    hero_path = "assets/arxora_logo_hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)
    else:
        st.markdown("""
        <div style="border-radius:8px;overflow:hidden;
                    box-shadow:0 0 0 1px rgba(0,0,0,.06),0 12px 32px rgba(0,0,0,.18);">
          <div style="background:#5B5BF7;padding:28px 16px;">
            <div style="max-width:1120px;margin:0 auto;">
              <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
                          color:#fff;font-weight:700;letter-spacing:.4px;
                          font-size:clamp(36px,7vw,72px);line-height:1.05;">
                Arxora
              </div>
            </div>
          </div>
          <div style="background:#000;padding:12px 16px 16px 16px;">
            <div style="max-width:1120px;margin:0 auto;">
              <div style="color:#fff;font-size:clamp(16px,2.4vw,28px);opacity:.92;">trade smarter.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

render_arxora_header()

# ===== Optional features (safe imports) =====
try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

# ===== Config / Helpers =====
# Единая политика TTL для снимков (часы)
def _ttl_hours_default() -> int:
    v = os.getenv("ARXORA_TTL_HOURS")
    if v and str(v).isdigit():
        return int(v)
    return 24  # дефолт — 24 часа

ETF_HINTS  = {"SPY","QQQ","IWM","DIA","EEM","EFA","XLK","XLF","XLE","XLY","XLI","XLV","XLP","XLU","VNQ","GLD","SLV"}
ENTRY_MARKET_EPS = float(os.getenv("ARXORA_ENTRY_MARKET_EPS", "0.0015"))
MIN_TP_STEP_PCT  = float(os.getenv("ARXORA_MIN_TP_STEP_PCT", "0.0010"))

def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _add_hours_iso(iso_utc: str, hours: int) -> str:
    dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (dt + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

def _fmt(x: Any) -> str:
    try: return f"{float(x):.2f}"
    except Exception: return "0.00"

def compute_risk_pct(levels: Dict[str, float]) -> str:
    entry = float(levels.get("entry", 0)); sl = float(levels.get("sl", 0))
    return "—" if entry == 0 else f"{abs(entry - sl)/max(1e-9,abs(entry))*100.0:.1f}"

def normalize_for_polygon(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(" ", "")
    if s.startswith(("X:", "C:", "O:")):
        head, tail = s.split(":", 1)
        tail = tail.replace("USDT", "USD").replace("USDC", "USD")
        return f"{head}:{tail}"
    if re.match(r"^[A-Z]{2,10}USD(T|C)?$", s):
        s = s.replace("USDT", "USD").replace("USDC", "USD")
        return f"X:{s}"
    return s

def sanitize_targets(action: str, entry: float, tp1: float, tp2: float, tp3: float):
    step = max(MIN_TP_STEP_PCT * max(1.0, abs(entry)), 1e-6 * max(1.0, abs(entry)))
    if action == "BUY":
        a = sorted([tp1, tp2, tp3]); a[0]=max(a[0], entry+step); a[1]=max(a[1], a[0]+step); a[2]=max(a[2], a[1]+step); return a[0],a[1],a[2]
    if action == "SHORT":
        a = sorted([tp1, tp2, tp3], reverse=True); a[0]=min(a[0], entry-step); a[1]=min(a[1], a[0]-step); a[2]=min(a[2], a[1]-step); return a[0],a[1],a[2]
    return tp1, tp2, tp3

def entry_mode_labels(action: str, entry: float, last_price: float, eps: float):
    if action not in ("BUY", "SHORT"): return "WAIT", "Entry"
    if abs(entry - last_price) <= eps * max(1.0, abs(last_price)): return "Market price", "Entry (Market)"
    if action == "BUY":  return ("Buy Stop","Entry (Buy Stop)") if entry > last_price else ("Buy Limit","Entry (Buy Limit)")
    else:                return ("Sell Stop","Entry (Sell Stop)") if entry < last_price else ("Sell Limit","Entry (Sell Limit)")

def rr_line(levels: Dict[str, float]) -> str:
    risk = abs(levels["entry"] - levels["sl"])
    if risk <= 1e-9: return ""
    rr1 = abs(levels["tp1"] - levels["entry"]) / risk
    rr2 = abs(levels["tp2"] - levels["entry"]) / risk
    rr3 = abs(levels["tp3"] - levels["entry"]) / risk
    return f"RR ≈ 1:{rr1:.1f} (TP1) · 1:{rr2:.1f} (TP2) · 1:{rr3:.1f} (TP3)"

# ===== Dynamic strategy import =====
def _load_strategy_module():
    try:
        mod = importlib.import_module("core.strategy")
        try:
            mod = importlib.reload(mod)  # подхват свежих правок
        except Exception:
            pass
        return mod, None
    except Exception:
        return None, traceback.format_exc()

def get_available_models() -> List[str]:
    mod, _ = _load_strategy_module()
    if not mod: return []
    reg = getattr(mod, "STRATEGY_REGISTRY", {}) or {}
    keys = list(reg.keys())
    return (["Octopus"] if "Octopus" in keys else []) + [k for k in sorted(keys) if k != "Octopus"]

def run_model_via_decide(ticker_norm: str, model_name: str) -> Dict[str, Any]:
    mod, err = _load_strategy_module()
    if not mod:
        raise RuntimeError("Не удалось импортировать core.strategy:\n" + (err or ""))
    if hasattr(mod, "decide"):
        # Горизонт убран из UI: задаём константу
        return mod.decide(ticker_norm, model_name, "Кратко")
    # Fallback — прямые функции
    name = (model_name or "").lower()
    if name == "octopus" and hasattr(mod, "analyze_asset_octopus"):
        return mod.analyze_asset_octopus(ticker_norm, "Кратко")
    if name == "global" and hasattr(mod, "analyze_asset_global"):
        return mod.analyze_asset_global(ticker_norm, "Кратко")
    for fname in ("analyze_asset_octopus","analyze_asset_global","analyze_asset_m7","analyze_asset_w7","analyze_asset"):
        if hasattr(mod, fname):
            fn = getattr(mod, fname)
            try:    return fn(ticker_norm, "Кратко")
            except TypeError:
                return fn(ticker_norm)
    raise RuntimeError("Точка входа стратегий отсутствует в core.strategy.")

# ===== Local snapshots with valid_until =====
SNAP_STORE = Path("snapshots_store"); SNAP_STORE.mkdir(exist_ok=True)
SNAP_INDEX = SNAP_STORE / "index.csv"

def _hash_weights(weights: Dict[str, float] | None) -> str:
    if not weights: return "none"
    s = json.dumps(weights, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:8]

def save_snapshot_local(payload: Dict[str, Any]) -> str:
    gen_at = _now_iso()
    ttl_h  = _ttl_hours_default()
    valid_until = _add_hours_iso(gen_at, ttl_h)
    meta = {
        "generated_at": gen_at,
        "valid_until": valid_until,
        "ttl_hours": ttl_h,
        "ticker": payload.get("ticker"),
        "horizon": "Кратко",
        "model": payload.get("model"),
        "model_version": payload.get("model_version","v1"),
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
        "note_html": payload.get("note_html",""),
        "alt": payload.get("alt",""),
    }
    raw = {"meta": meta, "body": body}
    sid = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]
    raw["snapshot_id"] = sid
    (SNAP_STORE / f"{sid}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    if not SNAP_INDEX.exists():
        with open(SNAP_INDEX, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["snapshot_id","generated_at","valid_until","ttl_hours","ticker","horizon","model","model_version","weights_hash"])
    with open(SNAP_INDEX, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([sid, meta["generated_at"], meta["valid_until"], ttl_h, meta["ticker"], meta["horizon"], meta["model"], meta["model_version"], meta["weights_hash"]])
    return sid

def load_snapshot_local(sid: str) -> Dict[str, Any]:
    p = SNAP_STORE / f"{sid}.json"
    if not p.exists(): raise FileNotFoundError(f"Snapshot {sid} not found")
    return json.loads(p.read_text(encoding="utf-8"))

def compare_snapshots_local(a_id: str, b_id: str) -> Dict[str, Any]:
    a = load_snapshot_local(a_id); b = load_snapshot_local(b_id)
    def rec(x): return x["body"].get("recommendation", {}) or {}
    def lv(x):  return x["body"].get("levels", {}) or {}
    def act(x): return str(rec(x).get("action","WAIT"))
    def cf(x):  return float(rec(x).get("confidence", 0.5))
    delta = {
        "action": {"a": act(a), "b": act(b)},
        "confidence": {"a": cf(a), "b": cf(b), "diff": cf(b)-cf(a)},
        "levels": {"a": lv(a), "b": lv(b)},
        "meta": {"a": a["meta"], "b": b["meta"]},
    }
    return {"a": a, "b": b, "delta": delta}

# ===== Confidence breakdown (fallback) =====
try:
    from core.ui_confidence import render_confidence_breakdown_inline as _render_breakdown_native
    from core.ui_confidence import get_confidence_breakdown_from_session as _get_conf_from_session
except Exception:
    _render_breakdown_native = None
    _get_conf_from_session = None

def render_confidence_breakdown_inline(ticker: str, conf_pct: float):
    try:
        st.session_state["last_overall_conf_pct"] = float(conf_pct or 0.0)
        st.session_state.setdefault("last_rules_pct", 44.0)
    except Exception:
        pass
    try:
        if _render_breakdown_native:
            return _render_breakdown_native(ticker, float(conf_pct or 0.0))
    except Exception:
        pass
    data = _get_conf_from_session() if _get_conf_from_session else {
        "overall_confidence_pct": float(st.session_state.get("last_overall_conf_pct", conf_pct or 0.0)),
        "breakdown": {
            "rules_pct": float(st.session_state.get("last_rules_pct", 44.0)),
            "ai_override_delta_pct": float(st.session_state.get("last_overall_conf_pct", conf_pct or 0.0)) - float(st.session_state.get("last_rules_pct", 44.0))
        },
        "shap_top": []
    }
    st.markdown("#### Confidence breakdown")
    st.write(f"Общая уверенность: {data.get('overall_confidence_pct',0):.1f}%")
    b = data.get("breakdown", {})
    st.write(f"— Базовые правила: {b.get('rules_pct',0):.1f}%")
    st.write(f"— AI override: {b.get('ai_override_delta_pct',0):.1f}%")

# ===== Main UI =====
st.subheader("AI agents")

_models = get_available_models()
if not _models:
    _models = ["Octopus", "Global"]

model = st.radio("Выберите модель", options=_models, index=0, horizontal=False, key="agent_radio")

ticker_input = st.text_input(
    "Тикер",
    value="",
    placeholder="Примеры: AAPL • TSLA • X:BTCUSD • BTCUSDT • C:EURUSD • O:SPY240920C500",
    help="Поддерживаются: акции/ETF (AAPL), крипто (X:BTCUSD или BTCUSDT), FX (C:EURUSD), опционы (O:<...>)"
)
ticker = ticker_input.strip().upper()
symbol_for_engine = normalize_for_polygon(ticker)

run = st.button("Проанализировать", type="primary", key="main_analyze")
st.write(f"Mode: AI · Model: {model}")

if run and ticker:
    try:
        out = run_model_via_decide(symbol_for_engine, model)

        # Сохранить результат для Share/Compare вне перерисовок
        st.session_state["last_result"] = {
            "out": out, "ticker": ticker, "model": model
        }

        # Цена: аккуратно показать либо fallback «—»
        last_price = float(out.get("last_price", 0.0) or 0.0)
        price_text = ("—" if last_price <= 0 else f"${last_price:.2f}")
        st.markdown(f"<div style='font-size:3rem; font-weight:800; text-align:center; margin:6px 0 14px 0;'>{price_text}</div>", unsafe_allow_html=True)

        # Сигнал
        rec = dict(out.get("recommendation", {}))
        action = rec.get("action", "WAIT")
        conf = float(rec.get("confidence", 0))
        conf_pct_val = conf*100.0 if conf <= 1.0 else conf
        conf_pct = f"{int(round(conf_pct_val))}%"

        lv = {k: float(out.get("levels", {}).get(k, 0.0)) for k in ("entry","sl","tp1","tp2","tp3")}
        if action in ("BUY", "SHORT"):
            t1, t2, t3 = sanitize_targets(action, lv["entry"], lv["tp1"], lv["tp2"], lv["tp3"])
            lv["tp1"], lv["tp2"], lv["tp3"] = float(t1), float(t2), float(t3)

        mode_text, entry_title = entry_mode_labels(action, lv.get("entry", last_price), last_price, ENTRY_MARKET_EPS)
        header_text = "WAIT"
        if action == "BUY": header_text = f"Long • {mode_text}"
        elif action == "SHORT": header_text = f"Short • {mode_text}"

        st.markdown(f"""
        <div style="background:#c57b0a; padding:14px 16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); margin-bottom:10px;">
            <div style="font-size:1.15rem; font-weight:700;">{header_text}</div>
            <div style="opacity:0.75; font-size:0.95rem; margin-top:2px;">{conf_pct} confidence</div>
        </div>
        """, unsafe_allow_html=True)

        ttl_h = _ttl_hours_default()
        valid_until = _add_hours_iso(_now_iso(), ttl_h)
        st.caption(f"As‑of: {_now_iso()} UTC • Valid until: {valid_until} • Model: {model}")

        render_confidence_breakdown_inline(ticker, conf_pct_val)

        if action in ("BUY", "SHORT"):
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div style='margin-bottom:8px'></div>", unsafe_allow_html=True); st.markdown(card_html(entry_title, f"{lv['entry']:.2f}", color="green"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("Stop Loss", f"{lv['sl']:.2f}", color="red"), unsafe_allow_html=True)
            with c3: st.markdown(card_html("TP 1", f"{lv['tp1']:.2f}"), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: st.markdown(card_html("TP 2", f"{lv['tp2']:.2f}"), unsafe_allow_html=True)
            with c2: st.markdown(card_html("TP 3", f"{lv['tp3']:.2f}"), unsafe_allow_html=True)
            rr = rr_line(lv)
            if rr:
                st.markdown(f"<div style='margin-top:6px; color:#FFA94D; font-weight:600;'>{rr}</div>", unsafe_allow_html=True)

        # Короткий контекст (если есть)
        ctx = out.get("context", [])
        if ctx:
            st.caption(f"Context: {ctx[0] if isinstance(ctx, list) and ctx else str(ctx)}")

        # Мягкий перфоманс (если модуль есть)
        st.subheader(f"Эффективность модели {model} по ключевым инструментам (3 месяца)")
        cols = st.columns(2)
        for i, tk in enumerate(["SPY","QQQ","BTCUSD","ETHUSD"]):
            perf_data = None
            try: perf_data = get_agent_performance(model, tk)
            except Exception: perf_data = None
            with cols[i % 2]:
                st.markdown(f"**{tk}**")
                if perf_data is not None:
                    perf_data = perf_data.set_index('date')
                    st.line_chart(perf_data["cumulative_return"])
                else:
                    st.info("Данных пока нет")

    except Exception as e:
        st.error(f"Ошибка анализа: {e}")
        st.exception(e)

elif not ticker:
    st.info("Введите тикер в поле выше и нажмите «Проанализировать». Примеры формата показаны в placeholder.")

# ===== Share / Compare (всегда доступно, работает при наличии результата) =====
with st.expander("Share snapshot / Compare", expanded=False):
    lr = st.session_state.get("last_result")
    if not lr:
        st.info("Сначала выполните анализ — затем можно сохранить снимок или сравнить два снимка.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Share snapshot", key="btn_share_snapshot"):
                payload = {
                    "ticker": lr["ticker"],
                    "model": lr["model"],
                    "model_version": "v1",
                    "weights": lr["out"].get("weights", {}),
                    "data_window": lr["out"].get("data_window", {}),
                    "recommendation": lr["out"].get("recommendation"),
                    "levels": lr["out"].get("levels"),
                    "context": lr["out"].get("context", []),
                    "agents": lr["out"].get("agents", []),
                    "last_price": lr["out"].get("last_price", 0.0),
                    "note_html": lr["out"].get("note_html",""),
                    "alt": lr["out"].get("alt",""),
                }
                sid = save_snapshot_local(payload)
                st.success(f"Snapshot saved: {sid}")
                st.code(sid)
        with c2:
            a_id = st.text_input("Snapshot A ID")
            b_id = st.text_input("Snapshot B ID")
            if st.button("Compare", key="btn_compare"):
                try:
                    out_cmp = compare_snapshots_local(a_id.strip(), b_id.strip())
                    st.json(out_cmp["delta"])
                    meta_a = out_cmp["delta"]["meta"]["a"]; meta_b = out_cmp["delta"]["meta"]["b"]
                    st.write(f"A: as‑of {meta_a.get('generated_at')} UTC → valid until {meta_a.get('valid_until')} (TTL {meta_a.get('ttl_hours')}h)")
                    st.write(f"B: as‑of {meta_b.get('generated_at')} UTC → valid until {meta_b.get('valid_until')} (TTL {meta_b.get('ttl_hours')}h)")
                except Exception as e:
                    st.error(f"Compare error: {e}")

# ===== Footer =====
st.markdown("---")
st.markdown("""
<style>
    .stButton > button { font-weight: 600; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center; opacity:.75; font-size:.9rem; padding:8px 0;'>© Arxora. All rights reserved.</div>",
    unsafe_allow_html=True
)
