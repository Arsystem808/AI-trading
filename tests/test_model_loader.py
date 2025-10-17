"""
tests/test_model_loader.py — Unit tests for model loader
========================================================
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.model_loader import (
    normalize_symbol,
    load_model_for,
    clear_model_cache,
    get_cache_stats,
    _extract_feature_names_if_any,
    _validate_model_structure
)


class TestNormalizeSymbol:
    def test_crypto_with_prefix(self):
        assert normalize_symbol("X:BTCUSD") == "X:BTCUSD"
    
    def test_crypto_without_prefix(self):
        assert normalize_symbol("BTCUSD") == "X:BTCUSD"
    
    def test_forex_known(self):
        assert normalize_symbol("EURUSD") == "C:EURUSD"
        assert normalize_symbol("JPYUSD") == "C:JPYUSD"
    
    def test_index(self):
        assert normalize_symbol("SPX") == "I:SPX"
    
    def test_unknown_usd(self):
        # Should default to crypto
        result = normalize_symbol("XYZUSD")
        assert result == "X:XYZUSD"


class TestFeatureExtraction:
    def test_sklearn_wrapper(self):
        mock_model = MagicMock()
        mock_model.booster_ = MagicMock()
        mock_model.booster_.feature_name.return_value = ["feat1", "feat2"]
        
        features = _extract_feature_names_if_any(mock_model)
        assert features == ["feat1", "feat2"]
    
    def test_raw_booster(self):
        mock_model = MagicMock()
        mock_model.feature_name.return_value = ["feat1", "feat2", "feat3"]
        
        features = _extract_feature_names_if_any(mock_model)
        assert features == ["feat1", "feat2", "feat3"]
    
    def test_no_features(self):
        mock_model = MagicMock(spec=[])
        
        features = _extract_feature_names_if_any(mock_model)
        assert features is None


class TestModelValidation:
    def test_valid_model_with_predict(self):
        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        
        assert _validate_model_structure(mock_model, Path("test.pkl"))
    
    def test_valid_model_with_predict_proba(self):
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock()
        
        assert _validate_model_structure(mock_model, Path("test.pkl"))
    
    def test_invalid_model(self):
        mock_model = MagicMock(spec=[])
        
        assert not _validate_model_structure(mock_model, Path("test.pkl"))


class TestCacheManagement:
    def setup_method(self):
        clear_model_cache()
    
    def test_cache_stats_empty(self):
        stats = get_cache_stats()
        assert stats["size"] == 0
        assert stats["keys"] == []
    
    @patch("core.model_loader._load_model_impl")
    def test_cache_hit(self, mock_load):
        mock_model = {"model": MagicMock(), "feature_names": ["f1"]}
        mock_load.return_value = mock_model
        
        # First load
        result1 = load_model_for("BTCUSD", agent="arxora_m7pro")
        
        # Second load (should use cache)
        result2 = load_model_for("BTCUSD", agent="arxora_m7pro")
        
        # Should only load once
        assert mock_load.call_count == 1
        assert result1 is not None
        assert result2 is not None
    
    def test_clear_specific_cache(self):
        with patch("core.model_loader._load_model_impl") as mock_load:
            mock_load.return_value = {"model": MagicMock()}
            
            load_model_for("BTCUSD", agent="arxora_m7pro")
            load_model_for("ETHUSD", agent="octopus")
            
            stats = get_cache_stats()
            assert stats["size"] == 2
            
            clear_model_cache("BTCUSD", "arxora_m7pro")
            
            stats = get_cache_stats()
            assert stats["size"] == 1


class TestIntegration:
    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            configs_dir = Path(tmpdir) / "configs"
            models_dir.mkdir()
            configs_dir.mkdir()
            
            yield {
                "root": Path(tmpdir),
                "models": models_dir,
                "configs": configs_dir
            }
    
    def test_load_from_config(self, temp_workspace):
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        
        model_path = temp_workspace["models"] / "test_model.joblib"
        
        with patch("joblib.load", return_value=mock_model):
            # Create config
            config = {
                "model_artifact": str(model_path),
                "feature_cols": ["feat1", "feat2"],
                "version": "1.0"
            }
            
            config_path = temp_workspace["configs"] / "m7pro_X_BTCUSD.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
            
            # Mock file existence
            with patch.object(Path, "exists", return_value=True):
                with patch("core.model_loader.MODELS_DIR", temp_workspace["models"]):
                    with patch("core.model_loader.CONFIG_DIR", temp_workspace["configs"]):
                        result = load_model_for("BTCUSD", agent="m7pro")
            
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
