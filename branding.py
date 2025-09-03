
# branding.py — drop-in header for Arxora style (purple/black + logo)
from pathlib import Path
import streamlit as st

PURPLE = "#5B5BF7"
BLACK = "#0B0D0E"

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --purple: {PURPLE};
  --black: {BLACK};
}}
html, body, [class^='css'] {{ font-family: 'Manrope', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }}
.hero {{ border-radius: 8px; overflow: hidden; margin: 0 0 20px 0; box-shadow: 0 0 0 1px rgba(0,0,0,0.06), 0 12px 32px rgba(0,0,0,0.18);}}
.hero-top {{ background: var(--purple); padding: 30px 0 22px 0; }}
.hero-bottom {{ background: var(--black); padding: 14px 0 18px 0; }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 0 12px; }}
.logo-img {{ display:block; height: clamp(52px, 10vw, 100px);}}
.tagline {{ color: #fff; font-size: clamp(18px, 2.6vw, 28px); opacity: .92; }}
</style>
"""

def render_header(logo_path: str = "assets/arxora_logo.png", tagline: str = "trade smarter."):
    st.markdown(CSS, unsafe_allow_html=True)
    html = f'''
    <div class="hero">
      <div class="hero-top">
        <div class="wrap">
          <img class="logo-img" src="{logo_path}" alt="Arxora"/>
        </div>
      </div>
      <div class="hero-bottom">
        <div class="wrap">
          <span class="tagline">{tagline}</span>
        </div>
      </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)
