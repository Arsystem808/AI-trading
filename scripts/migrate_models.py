"""
scripts/migrate_models.py — Auto-generate JSON configs for existing models
===========================================================================

Scans models/ directory and creates corresponding JSON configs in configs/
with metadata extracted from model files.

Usage:
    python scripts/migrate_models.py [--dry-run] [--force]
    
    --dry-run: Show what would be created without writing files
    --force: Overwrite existing config files
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib

from core.model_loader import MODELS_DIR, CONFIG_DIR, _extract_feature_names_if_any
from core.utils_naming import parse_model_filename, generate_config_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_config_for_model(model_path: Path, force: bool = False) -> bool:
    """
    Generate JSON config for a model file.
    
    Args:
        model_path: Path to model file (.joblib/.pkl)
        force: Overwrite existing config
        
    Returns:
        bool: True if config was created
    """
    # Parse filename
    parsed = parse_model_filename(model_path.name)
    
    if not parsed["ticker"]:
        logger.warning(f"Could not parse ticker from {model_path.name}")
        return False
    
    agent = parsed["agent"] or "unknown"
    ticker = parsed["ticker"].replace("_", ":")  # X_BTCUSD -> X:BTCUSD
    
    # Generate config filename
    config_name = generate_config_filename(ticker, agent)
    config_path = CONFIG_DIR / config_name
    
    if config_path.exists() and not force:
        logger.info(f"Config exists: {config_path} (use --force to overwrite)")
        return False
    
    # Load model to extract metadata
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load {model_path}: {e}")
        return False
    
    # Extract features
    feature_names = _extract_feature_names_if_any(model)
    
    # Build config
    config = {
        "model_artifact": str(model_path.relative_to(Path("."))),
        "agent": agent,
        "ticker": ticker,
        "version": parsed["version"] or "1.0",
        "migrated_at": datetime.now().isoformat(),
        "migrated_by": "scripts/migrate_models.py"
    }
    
    if feature_names:
        config["feature_cols"] = feature_names
        logger.info(f"Extracted {len(feature_names)} features from {model_path.name}")
    
    # Check for scaler
    scaler_candidates = [
        model_path.parent / model_path.name.replace("_model", "_scaler").replace(".joblib", ".pkl"),
        model_path.parent / f"{agent}_scaler_{ticker.replace(':', '_')}.pkl"
    ]
    
    for scaler_path in scaler_candidates:
        if scaler_path.exists():
            config["scaler_artifact"] = str(scaler_path.relative_to(Path(".")))
            logger.info(f"Found scaler: {scaler_path.name}")
            break
    
    # Write config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Created: {config_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate JSON configs for existing models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created")
    parser.add_argument("--force", action="store_true", help="Overwrite existing configs")
    args = parser.parse_args()
    
    if not MODELS_DIR.exists():
        logger.error(f"Models directory not found: {MODELS_DIR}")
        return
    
    # Find all model files
    model_files = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.pkl"))
    
    # Exclude scalers
    model_files = [f for f in model_files if "scaler" not in f.name.lower()]
    
    logger.info(f"Found {len(model_files)} model files in {MODELS_DIR}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be created")
    
    created_count = 0
    
    for model_path in sorted(model_files):
        logger.info(f"\nProcessing: {model_path.name}")
        
        if args.dry_run:
            parsed = parse_model_filename(model_path.name)
            logger.info(f"  Agent: {parsed['agent']}")
            logger.info(f"  Ticker: {parsed['ticker']}")
            logger.info(f"  Version: {parsed['version']}")
            continue
        
        if generate_config_for_model(model_path, force=args.force):
            created_count += 1
    
    logger.info(f"\n{'Would create' if args.dry_run else 'Created'} {created_count} config(s)")


if __name__ == "__main__":
    main()
