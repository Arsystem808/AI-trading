# utils/llm.py
import os
import streamlit as st
from openai import OpenAI

def get_client():
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None

def get_model(default="gpt-4o-mini"):
    return st.secrets.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or default

def chat(system: str, user: str, temperature=0.2, max_tokens=320) -> str | None:
    client = get_client()
    if not client:
        return None
    res = client.chat.completions.create(
        model=get_model(),
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (res.choices[0].message.content or "").strip()
