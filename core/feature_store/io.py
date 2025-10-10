# core/feature_store/io.py
from feast import FeatureStore

def get_store(repo_path: str = "core/feature_store/feature_repo"):
    return FeatureStore(repo_path=repo_path)
