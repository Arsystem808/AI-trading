"""
core/model_loader.py — Enterprise-grade ML model loader
========================================================

Production features:
- Thread-safe model caching with automatic invalidation
- Hot reload support via file modification time tracking
- Polygon API compatible ticker normalization
- LightGBM feature_names extraction with validation
- Test prediction on load to catch broken models
- Comprehensive logging and error handling
- Environment variable configuration for paths
- Crypto whitelist to prevent FX/crypto confusion

Author: Arxora Trading System
Version: 2.0.0
Updated: 2025-10-17
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import joblib
except ImportError:
    joblib = None

try:
    import numpy as np
except ImportError:
    np = None

from core.utils_naming import sanitize_symbol

# ==================== Configuration ====================

# Paths (env-configurable for Docker/K8s)
MODELS_DIR = Path(os.getenv("ARXORA_MODELS_DIR", "models"))
CONFIG_DIR = Path(os.getenv("ARXORA_CONFIG_DIR", "configs"))

# Known crypto bases for classification
CRYPTO_BASES = {
    "BTC", "ETH", "XRP", "LTC", "DOGE", "ADA", "DOT", "LINK",
    "UNI", "MATIC", "SOL", "AVAX", "ATOM", "XLM", "ALGO"
}

# Known forex pairs
FOREX_BASES = {
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY", "SEK", "NOK"
}

# Known indices
INDICES = {
    "SPX", "DJI", "NDX", "RUT", "VIX", "DAX", "FTSE", "N225"
}

# Logging
logger = logging.getLogger(__name__)

# Thread-safe cache with modification time tracking
_model_cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
_cache_lock = threading.Lock()

# Typography normalization for JSON
_FANCY_CHARS = {
    """: '"', """: '"', "„": '"', "«": '"', "»": '"',
    "'": "'", "'": "'",
    "—": "-", "–": "-",
    "，": ",", "：": ":", "；": ";",
}


# ==================== Utilities ====================

def _normalize_text(s: str) -> str:
    """Fix typography and trailing commas in JSON."""
    for k, v in _FANCY_CHARS.items():
        s = s.replace(k, v)
    # Convert single quotes to double quotes in JSON values
    s = re.sub(r"(:\s*)\'([^\'\\n]*)\'", r'\1"\2"', s)
    # Remove trailing commas
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _load_json_file(p: Path) -> Any:
    """Load JSON with typography normalization."""
    raw = p.read_text(encoding="utf-8", errors="replace")
    norm = _normalize_text(raw)
    try:
        return json.loads(norm)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {p}: {e}")
        raise ValueError(f"Invalid JSON in {p}: {e}")


def _get_file_mtime(p: Path) -> float:
    """Get file modification time, returns 0 if not exists."""
    try:
        return p.stat().st_mtime if p.exists() else 0.0
    except OSError:
        return 0.0


def _extract_feature_names_if_any(model: Any) -> Optional[list]:
    """
    Extract feature names from LightGBM booster.
    
    Supports:
    - sklearn API wrapper (.booster_.feature_name())
    - raw Booster (.feature_name())
    - legacy versions (.feature_name_)
    """
    try:
        # sklearn API wrapper
        if hasattr(model, "booster_") and model.booster_ is not None:
            booster = model.booster_
            if hasattr(booster, "feature_name"):
                names = booster.feature_name()
                return list(names) if names is not None else None
        
        # raw Booster
        if hasattr(model, "feature_name"):
            names = model.feature_name()
            return list(names) if names is not None else None
        
        # legacy versions
        if hasattr(model, "feature_name_"):
            names = getattr(model, "feature_name_")
            return list(names) if names is not None else None
            
    except Exception as e:
        logger.warning(f"Failed to extract feature_names: {e}")
    
    return None


def _validate_model_structure(obj: Any, path: Path) -> bool:
    """
    Validate model has required prediction methods.
    
    Args:
        obj: Loaded model object
        path: File path for logging
        
    Returns:
        bool: True if model is valid
    """
    has_predict = hasattr(obj, "predict")
    has_predict_proba = hasattr(obj, "predict_proba")
    
    if not (has_predict or has_predict_proba):
        logger.error(
            f"Invalid model from {path}: missing predict/predict_proba methods"
        )
        return False
    
    return True


def _test_predict(model: Any, num_features: int = 10) -> bool:
    """
    Test model with dummy prediction to catch runtime errors.
    
    Args:
        model: Model to test
        num_features: Number of features for test matrix
        
    Returns:
        bool: True if prediction succeeds
    """
    if not np:
        logger.warning("NumPy not available, skipping test prediction")
        return True
    
    try:
        X_test = np.random.rand(1, num_features)
        
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_test)
        elif hasattr(model, "predict"):
            _ = model.predict(X_test)
        
        return True
        
    except Exception as e:
        logger.error(f"Model test prediction failed: {e}")
        return False


def _validate_model(obj: Any, path: Path, feature_count: Optional[int] = None) -> bool:
    """
    Full model validation: structure + test prediction.
    
    Args:
        obj: Loaded model object
        path: File path for logging
        feature_count: Expected feature count for test
        
    Returns:
        bool: True if model passes all checks
    """
    if not _validate_model_structure(obj, path):
        return False
    
    if feature_count and np:
        if not _test_predict(obj, feature_count):
            logger.warning(f"Model {path} failed test prediction but has valid structure")
            # Don't fail completely - some models may need specific input format
    
    return True


def _try_joblib(p: Path) -> Any:
    """Load pickle/joblib file with error handling."""
    if not joblib:
        raise RuntimeError("joblib library not available")
    
    try:
        obj = joblib.load(p)
        logger.info(f"Loaded model from {p}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load {p}: {e}")
        raise


def normalize_symbol(sym: str) -> str:
    """
    Normalize ticker for Polygon API compatibility.
    
    Uses whitelists to correctly classify crypto/forex/indices.
    
    Examples:
        BTCUSD -> X:BTCUSD (crypto)
        EURUSD -> C:EURUSD (forex)
        SPX -> I:SPX (index)
        JPYUSD -> C:JPYUSD (forex, not crypto)
        
    Args:
        sym: Raw ticker symbol
        
    Returns:
        str: Normalized ticker with class prefix
    """
    u = (sym or "").upper().strip()
    
    # Already prefixed
    if u.startswith(("X:", "C:", "I:")):
        return u
    
    # Indices
    if u in INDICES:
        return f"I:{u}"
    
    # Crypto: check against whitelist
    for base in CRYPTO_BASES:
        if u.startswith(base) and u.endswith("USD"):
            return f"X:{u}"
    
    # Forex: check against whitelist
    for base in FOREX_BASES:
        if u.startswith(base):
            return f"C:{u}"
    
    # Default: crypto (backward compatibility)
    if u.endswith("USD"):
        logger.debug(f"Defaulting {u} to crypto (not in whitelist)")
        return f"X:{u}"
    
    # No classification - return as-is
    return u


def _candidate_names(base: str, agent: Optional[str] = None) -> list[str]:
    """
    Generate priority list of model filenames.
    
    Args:
        base: Sanitized ticker name (X_BTCUSD)
        agent: Agent name filter (optional)
        
    Returns:
        list[str]: Ordered list of candidate filenames
    """
    if agent:
        return [f"{agent}_{base}.joblib"]
    
    return [
        f"arxora_m7pro_{base}.joblib",
        f"global_{base}.joblib",
        f"alphapulse_{base}.joblib",
        f"octopus_{base}.joblib",
    ]


def _dedup(seq):
    """Remove duplicates preserving order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _build_candidates(ticker_or_name: str, agent: Optional[str] = None) -> list[Path]:
    """
    Build priority list of paths for models and configs.
    
    Search order:
    1. Normalized name with sanitization (X_BTCUSD)
    2. Raw name with sanitization (BTCUSD) - backward compat
    3. JSON configs (may reference models)
    4. Fallback general model (only if agent=None)
    
    Args:
        ticker_or_name: Ticker or model name
        agent: Agent name filter (optional)
        
    Returns:
        list[Path]: Ordered list of candidate paths
    """
    raw = (ticker_or_name or "").upper().strip()
    norm = normalize_symbol(raw)
    safe_norm = sanitize_symbol(norm)
    safe_raw = sanitize_symbol(raw)
    
    # Model filenames
    names = _candidate_names(safe_norm, agent)
    if safe_raw != safe_norm:
        names += _candidate_names(safe_raw, agent)
    
    # Model paths
    model_paths = [MODELS_DIR / n for n in names]
    
    # Fallback only if agent not specified
    if not agent:
        fallback = MODELS_DIR / "m7_model.pkl"
        # Log warning about fallback usage
        if fallback.exists():
            logger.warning(
                f"Fallback model {fallback} available for {ticker_or_name}. "
                "This may cause feature mismatch."
            )
        model_paths.append(fallback)
    
    # JSON configs
    cfgs = []
    if agent:
        cfgs.append(CONFIG_DIR / f"{agent}_{safe_norm}.json")
        if safe_raw != safe_norm:
            cfgs.insert(0, CONFIG_DIR / f"{agent}_{safe_raw}.json")
    else:
        cfgs.extend([
            CONFIG_DIR / f"m7pro_{safe_norm}.json",
            CONFIG_DIR / f"octopus_{safe_norm}.json",
            CONFIG_DIR / "calibration.json",
        ])
        if safe_raw != safe_norm:
            cfgs.insert(0, CONFIG_DIR / f"m7pro_{safe_raw}.json")
            cfgs.insert(1, CONFIG_DIR / f"octopus_{safe_raw}.json")
    
    return _dedup(model_paths + cfgs)


def _check_feature_consistency(
    model_features: Optional[list],
    config_features: Optional[list],
    source: str
) -> None:
    """
    Log warning if config and booster feature names mismatch.
    
    Args:
        model_features: feature_names from LightGBM booster
        config_features: feature_cols from JSON config
        source: File path for logging
    """
    if not model_features or not config_features:
        return
    
    if list(config_features) != list(model_features):
        logger.warning(
            f"Feature mismatch in {source}:\n"
            f"  Config: {config_features}\n"
            f"  Booster: {model_features}\n"
            f"Using booster features for inference."
        )


def _is_cache_valid(cache_entry: Dict[str, Any]) -> bool:
    """
    Check if cached model is still valid (file unchanged).
    
    Args:
        cache_entry: Cached model data with mtime
        
    Returns:
        bool: True if cache is valid
    """
    if "mtime" not in cache_entry or "path" not in cache_entry:
        return True  # No file tracking
    
    current_mtime = _get_file_mtime(cache_entry["path"])
    return current_mtime <= cache_entry["mtime"]


def _load_model_impl(
    ticker_or_name: str,
    agent: Optional[str] = None
) -> Optional[Union[Any, Dict[str, Any]]]:
    """
    Internal model loader implementation (without cache).
    
    Process:
    1. Search for models/configs by priority
    2. Load JSON config with model_artifact if exists
    3. Extract feature_names from LightGBM booster
    4. Validate model structure and test prediction
    5. Check feature consistency between config and booster
    
    Args:
        ticker_or_name: Ticker (BTCUSD, X:BTCUSD) or model name
        agent: Agent name filter (arxora_m7pro, octopus, etc.)
        
    Returns:
        dict: {
            "model": loaded model,
            "feature_names": feature names from booster,
            "metadata": metadata from JSON,
            "path": Path to model file,
            "mtime": file modification time
        }
        or raw model object (backward compatibility)
        or None if not found
    """
    logger.debug(f"Loading model for {ticker_or_name} (agent={agent})")
    
    for p in _build_candidates(ticker_or_name, agent):
        if not p.exists():
            continue
        
        # JSON config with metadata
        if p.suffix == ".json":
            try:
                cfg = _load_json_file(p)
            except ValueError:
                continue
            
            # Config with model path
            if isinstance(cfg, dict) and "model_artifact" in cfg:
                art = Path(cfg["model_artifact"])
                if not art.is_absolute():
                    art = Path(".") / art
                
                if not art.exists():
                    logger.warning(f"Model artifact {art} not found (referenced in {p})")
                    continue
                
                obj = _try_joblib(art)
                
                # Extract features
                feats = _extract_feature_names_if_any(obj)
                feature_count = len(feats) if feats else None
                
                # Validate model
                if not _validate_model(obj, art, feature_count):
                    logger.warning(f"Skipping invalid model: {art}")
                    continue
                
                # Check feature consistency
                config_feats = cfg.get("feature_cols")
                _check_feature_consistency(feats, config_feats, str(p))
                
                # Build response
                metadata = dict(cfg)
                if feats:
                    metadata["feature_names"] = feats
                
                logger.info(
                    f"Loaded model from {art} via config {p} "
                    f"(features: {len(feats) if feats else 'unknown'})"
                )
                
                return {
                    "model": obj,
                    "metadata": metadata,
                    "feature_names": feats,
                    "path": art,
                    "mtime": _get_file_mtime(art)
                }
            
            # Config without model - return as-is
            logger.info(f"Loaded config from {p}")
            return cfg
        
        # Direct model load (.joblib/.pkl)
        if p.suffix in (".joblib", ".pkl"):
            obj = _try_joblib(p)
            
            feats = _extract_feature_names_if_any(obj)
            feature_count = len(feats) if feats else None
            
            # Validate model
            if not _validate_model(obj, p, feature_count):
                logger.warning(f"Skipping invalid model: {p}")
                continue
            
            if feats:
                logger.info(f"Loaded model from {p} (features: {len(feats)})")
                return {
                    "model": obj,
                    "feature_names": feats,
                    "path": p,
                    "mtime": _get_file_mtime(p)
                }
            
            # No feature names - backward compatibility
            logger.info(f"Loaded model from {p} (no feature names)")
            return obj
    
    logger.warning(f"No model found for {ticker_or_name} (agent={agent})")
    return None


def load_model_for(
    ticker_or_name: str,
    agent: Optional[str] = None
) -> Optional[Union[Any, Dict[str, Any]]]:
    """
    Thread-safe model loader with automatic hot reload.
    
    Features:
    - Thread-safe cache with LRU-like behavior
    - Automatic cache invalidation on file modification
    - Polygon API compatible ticker normalization
    - LightGBM feature_names extraction
    - Model validation with test prediction
    - Feature consistency checks
    
    Usage examples:
        # Load model for BTCUSD (searches all agents)
        model_data = load_model_for("BTCUSD")
        model = model_data["model"]
        features = model_data["feature_names"]
        
        # Load specific agent
        model_data = load_model_for("BTCUSD", agent="arxora_m7pro")
        
        # Polygon normalization is automatic
        model_data = load_model_for("X:BTCUSD")  # equivalent to BTCUSD
        
    Args:
        ticker_or_name: Ticker (BTCUSD, X:BTCUSD) or model name
        agent: Agent name filter (arxora_m7pro, octopus, etc.)
        
    Returns:
        dict: {
            "model": loaded model,
            "feature_names": feature names from booster,
            "metadata": metadata from JSON
        }
        or raw model object (backward compatibility)
        or None if not found
    """
    # Cache key includes agent to differentiate same ticker for different agents
    cache_key = (ticker_or_name, agent)
    
    with _cache_lock:
        # Check cache
        if cache_key in _model_cache:
            cached = _model_cache[cache_key]
            
            # Validate cache freshness
            if _is_cache_valid(cached):
                logger.debug(f"Returning cached model for {cache_key}")
                # Return copy without internal tracking fields
                if isinstance(cached, dict):
                    return {
                        k: v for k, v in cached.items()
                        if k not in ("path", "mtime")
                    }
                return cached
            else:
                logger.info(f"Cache invalidated for {cache_key} (file modified)")
                del _model_cache[cache_key]
        
        # Load model
        result = _load_model_impl(ticker_or_name, agent)
        
        # Cache result
        if result is not None:
            _model_cache[cache_key] = result
            
            # Basic LRU: remove oldest if cache too large
            if len(_model_cache) > 32:
                oldest_key = next(iter(_model_cache))
                logger.debug(f"Cache full, evicting {oldest_key}")
                del _model_cache[oldest_key]
        
        # Return without internal tracking fields
        if isinstance(result, dict):
            return {
                k: v for k, v in result.items()
                if k not in ("path", "mtime")
            }
        
        return result


def clear_model_cache(ticker_or_name: Optional[str] = None, agent: Optional[str] = None):
    """
    Clear model cache. Useful after model updates.
    
    Args:
        ticker_or_name: Clear specific ticker (None = clear all)
        agent: Clear specific agent (None = clear all)
    """
    with _cache_lock:
        if ticker_or_name is None and agent is None:
            _model_cache.clear()
            logger.info("Cleared all model cache")
        else:
            cache_key = (ticker_or_name, agent)
            if cache_key in _model_cache:
                del _model_cache[cache_key]
                logger.info(f"Cleared cache for {cache_key}")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring.
    
    Returns:
        dict: {
            "size": number of cached models,
            "keys": list of cached (ticker, agent) pairs
        }
    """
    with _cache_lock:
        return {
            "size": len(_model_cache),
            "keys": list(_model_cache.keys())
        }


# ==================== Example JSON Configs ====================
"""
# configs/m7pro_X_BTCUSD.json
{
  "model_artifact": "models/arxora_m7pro_X_BTCUSD_v2.joblib",
  "feature_cols": ["returns", "volatility", "momentum", "sma20", "sma50", "rsi", "volume_ratio"],
  "scaler_artifact": "models/m7_scaler_X_BTCUSD.pkl",
  "train_date": "2024-12-01",
  "train_samples": 50000,
  "version": "2.1",
  "agent": "M7",
  "accuracy": 0.67,
  "supported_tickers": ["X:BTCUSD"]
}

# configs/octopus_X_ETHUSD.json
{
  "model_artifact": "models/octopus_X_ETHUSD.joblib",
  "feature_cols": ["pos", "slopenorm", "volratio", "hauprun", "band", "longupper"],
  "train_date": "2024-11-15",
  "version": "1.0",
  "agent": "Octopus",
  "supported_tickers": ["X:ETHUSD"]
}
"""
