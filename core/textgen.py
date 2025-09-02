from __future__ import annotations
from typing import Dict, Any
import random

def fmt_price(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")

def confidence_to_words(c: float) -> str:
    if c >= 0.8: return "высокая"
    if c >= 0.65: return "выше среднего"
    if c >= 0.5: return "средняя"
    if c >= 0.4: return "ниже среднего"
    return "низкая"

MOODS = [
    "Картина простая: импульс выдохся у «потолка».",
    "Движение замедляется — покупатели берут паузу.",
    "Цена возле ключевой границы: обычно здесь рынок делает передышку.",
    "Снизу близкая опора, сверху — давление. Баланс хрупкий.",
    "Последняя волна была резкой — логично ждать коррекцию/перезагрузку."
]

ALT_LEADS = [
    "Альтернативно:",
    "План Б:",
    "Если рынок дёрнет в другую сторону:",
    "Консервативно можно так:"
]

def build_narrative(symbol: str, horizon_name: str, rec: Dict[str, Any]) -> str:
    c_words = confidence_to_words(rec.get("confidence", 0.5))
    lines = []
    lines.append(f"**{symbol} — {horizon_name}**")
    lines.append("")
    lines.append(random.choice(MOODS))
    lines.append("")
    lines.append("**Рекомендация:** " + rec['action'])
    if rec.get("entry"):
        lines.append(f"• Вход: {fmt_price(rec['entry'])}")
    if rec.get("tp1"):
        lines.append(f"• Цель 1: {fmt_price(rec['tp1'])}")
    if rec.get("tp2"):
        lines.append(f"• Цель 2: {fmt_price(rec['tp2'])}")
    if rec.get("sl"):
        lines.append(f"• Стоп: {fmt_price(rec['sl'])}")
    lines.append(f"• Уверенность: {c_words} ({rec.get('confidence', 0.5):.2f})")
    lines.append("")
    lines.append("**" + random.choice(ALT_LEADS) + "** " + rec.get("alt_action","WAIT"))
    if rec.get("alt_entry") or rec.get("alt_tp1") or rec.get("alt_tp2") or rec.get("alt_sl"):
        if rec.get("alt_entry"):
            lines.append(f"• Альт. вход: {fmt_price(rec['alt_entry'])}")
        if rec.get("alt_tp1"):
            lines.append(f"• Альт. цель 1: {fmt_price(rec['alt_tp1'])}")
        if rec.get("alt_tp2"):
            lines.append(f"• Альт. цель 2: {fmt_price(rec['alt_tp2'])}")
        if rec.get("alt_sl"):
            lines.append(f"• Альт. стоп: {fmt_price(rec['alt_sl'])}")
    if rec.get("alt_note"):
        lines.append("— " + rec["alt_note"])
    lines.append("")
    if rec.get("commentary"):
        lines.append("**Комментарий:** " + rec["commentary"])
    lines.append("")
    lines.append("_Это не инвестиционная рекомендация. Решения вы принимаете самостоятельно._")
    return "\n".join(lines)
