# core/utils_naming.py
import re

def sanitize_symbol(s: str) -> str:
    s = s.upper().strip()
    return re.sub(r'[^A-Z0-9._-]+', '_', s).strip('_')
