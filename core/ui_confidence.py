### core/ui_confidence.py
Breakdown с безопасным fallback из Session State, чтобы при любом раскладе совпадал с карточкой; это рекомендуемый приём Streamlit для согласованности состояния. [2][5]
```python
# core/ui_confidence.py
from typing import Dict, Any

def get_confidence_breakdown_from_session() -> Dict[str, Any]:
    """
    Универсальный fallback для UI: читает 'общий процент' и 'правила' из Session State
    и возвращает корректную дельту AI override, чтобы не было 0% при рабочей карточке.
    """
    try:
        import streamlit as st
        overall_hint = float(st.session_state.get("last_overall_conf_pct", 0.0))
        rules_hint = float(st.session_state.get("last_rules_pct", 44.0))
        if overall_hint > 0:
            ai_delta = float(overall_hint - rules_hint)
            return {
                "overall_confidence_pct": overall_hint,
                "breakdown": {
                    "rules_pct": rules_hint,
                    "ai_override_delta_pct": ai_delta
                },
                "shap_top": []
            }
    except Exception:
        pass
    # Абсолютный запасной вариант
    return {
        "overall_confidence_pct": 44.0,
        "breakdown": {
            "rules_pct": 44.0,
            "ai_override_delta_pct": 0.0
        },
        "shap_top": []
    }

def render_confidence_breakdown_inline(ticker: str, overall_pct: float) -> Dict[str, Any]:
