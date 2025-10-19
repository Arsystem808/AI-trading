"""
core/utils_naming.py — Naming utilities for file-safe identifiers

Provides consistent naming conventions across the system for:
- Ticker symbol sanitization (Polygon API -> filesystem)
- Model artifact naming
- Config file naming

Author: Arxora Trading System
Version: 1.0.0
"""

import re
from typing import Optional


def sanitize_symbol(sym: str) -> str:
    """
    Convert ticker symbol to filesystem-safe name.
    
    Rules:
    - Colons (:) -> underscores (_) for Polygon prefixes
    - Slashes (/) -> hyphens (-) for alternatives
    - Spaces/backslashes/pipes -> underscores
    - Preserve alphanumeric and existing underscores/hyphens
    
    Examples:
        X:BTCUSD -> X_BTCUSD
        BTC/USD -> BTC-USD
        SPX 500 -> SPX_500
        
    Args:
        sym: Raw ticker symbol
        
    Returns:
        str: Filesystem-safe sanitized name
    """
    if not sym:
        return ""
    
    sanitized = (
        sym.replace(":", "_")
           .replace("/", "-")
           .replace(" ", "_")
           .replace("\\", "_")
           .replace("|", "_")
           .replace("?", "")
           .replace("*", "")
           .replace("<", "")
           .replace(">", "")
           .replace('"', "")
    )
    
    # Remove consecutive underscores/hyphens
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized)
    
    # Strip leading/trailing special chars
    sanitized = sanitized.strip("_-")
    
    return sanitized


def generate_model_filename(
    ticker: str,
    agent: str = "arxora_m7pro",
    version: Optional[str] = None,
    extension: str = ".joblib"
) -> str:
    """
    Generate standardized model filename.
    
    Format: {agent}_{sanitized_ticker}[_v{version}]{extension}
    
    Examples:
        ("BTCUSD", "arxora_m7pro") -> "arxora_m7pro_X_BTCUSD.joblib"
        ("X:ETHUSD", "octopus", "2.1") -> "octopus_X_ETHUSD_v2_1.joblib"
        
    Args:
        ticker: Raw ticker symbol
        agent: Agent name
        version: Model version (optional)
        extension: File extension (default .joblib)
        
    Returns:
        str: Standardized filename
    """
    from core.model_loader import normalize_symbol
    
    norm_ticker = normalize_symbol(ticker)
    safe_ticker = sanitize_symbol(norm_ticker)
    
    parts = [agent, safe_ticker]
    
    if version:
        # Sanitize version (replace dots with underscores)
        safe_version = version.replace(".", "_")
        parts.append(f"v{safe_version}")
    
    filename = "_".join(parts) + extension
    return filename


def generate_config_filename(
    ticker: str,
    agent: str = "m7pro",
    extension: str = ".json"
) -> str:
    """
    Generate standardized config filename.
    
    Format: {agent}_{sanitized_ticker}{extension}
    
    Examples:
        ("BTCUSD", "m7pro") -> "m7pro_X_BTCUSD.json"
        ("X:ETHUSD", "octopus") -> "octopus_X_ETHUSD.json"
        
    Args:
        ticker: Raw ticker symbol
        agent: Agent name
        extension: File extension (default .json)
        
    Returns:
        str: Standardized filename
    """
    from core.model_loader import normalize_symbol
    
    norm_ticker = normalize_symbol(ticker)
    safe_ticker = sanitize_symbol(norm_ticker)
    
    filename = f"{agent}_{safe_ticker}{extension}"
    return filename


def parse_model_filename(filename: str) -> dict:
    """
    Parse model filename into components.
    
    Examples:
        "arxora_m7pro_X_BTCUSD_v2_1.joblib" -> {
            "agent": "arxora_m7pro",
            "ticker": "X_BTCUSD",
            "version": "2.1",
            "extension": ".joblib"
        }
        
    Args:
        filename: Model filename
        
    Returns:
        dict: Parsed components
    """
    import os
    
    name, ext = os.path.splitext(filename)
    parts = name.split("_")
    
    result = {
        "agent": None,
        "ticker": None,
        "version": None,
        "extension": ext
    }
    
    # Try to identify agent (first part before ticker)
    known_agents = ["arxora", "m7pro", "global", "alphapulse", "octopus"]
    
    agent_parts = []
    ticker_parts = []
    version_parts = []
    
    in_version = False
    in_ticker = False
    
    for part in parts:
        if part.startswith("v") and part[1:].replace("_", ".").replace(".", "").isdigit():
            # Version indicator
            in_version = True
            version_parts.append(part[1:])
        elif in_version:
            version_parts.append(part)
        elif part in known_agents or (not in_ticker and not agent_parts):
            agent_parts.append(part)
        else:
            in_ticker = True
            ticker_parts.append(part)
    
    if agent_parts:
        result["agent"] = "_".join(agent_parts)
    if ticker_parts:
        result["ticker"] = "_".join(ticker_parts)
    if version_parts:
        result["version"] = ".".join(version_parts)
    
    return result
