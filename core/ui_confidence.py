# core/ui_confidence.py
<<<<<<< HEAD
from typing import Any, Dict

=======
# Безопасный Breakdown из Session State; при отсутствии/ошибках возвращает согласованный fallback.

from typing import Dict, Any
>>>>>>> origin/main


def _get_state_float(key: str, default: float) -> float:
    """
    Безопасно читает число из st.session_state (если Streamlit доступен), иначе отдаёт default.
    """
    try:
        import streamlit as st  # type: ignore

        return float(st.session_state.get(key, default))
    except Exception:
        return float(default)


def get_confidence_breakdown_from_session() -> Dict[str, Any]:
    """
    Универсальный fallback для UI: читает «общий процент» и «правила» из Session State
    и корректно вычисляет AI override delta; если подсказок нет, возвращает стабильный шаблон.
    """
    overall_hint = _get_state_float("last_overall_conf_pct", 0.0)
    rules_hint = _get_state_float("last_rules_pct", 44.0)

    if overall_hint > 0.0:
        ai_delta = float(overall_hint - rules_hint)
        return {
            "overall_confidence_pct": float(overall_hint),
            "breakdown": {
                "rules_pct": float(rules_hint),
                "ai_override_delta_pct": ai_delta,
            },
            "shap_top": [],
        }

    # Абсолютный запасной вариант, согласованный с карточкой
    return {
        "overall_confidence_pct": 44.0,
        "breakdown": {"rules_pct": 44.0, "ai_override_delta_pct": 0.0},
        "shap_top": [],
    }


def render_confidence_breakdown_inline(
    ticker: str, overall_pct: float
) -> Dict[str, Any]:
    """
    Возвращает breakdown для инлайн‑рендера; обновляет Session State, если Streamlit доступен.
    """
    if overall_pct <= 0:
        return get_confidence_breakdown_from_session()

    rules_hint = _get_state_float("last_rules_pct", 44.0)
    ai_delta = float(overall_pct - rules_hint)

    # Пытаемся записать подсказку обратно в Session State (не обязателен)
    try:
        import streamlit as st  # type: ignore

        st.session_state["last_overall_conf_pct"] = float(overall_pct)
        st.session_state.setdefault("last_rules_pct", float(rules_hint))
    except Exception:
        pass

    return {
        "overall_confidence_pct": float(overall_pct),
        "breakdown": {
            "rules_pct": float(rules_hint),
            "ai_override_delta_pct": ai_delta,
        },
        "shap_top": [],
    }
