import os
from huggingface_hub import HfApi

repo_id   = os.environ.get("HF_REPO", "org-or-user/nightly-models")
revision  = os.environ.get("HF_REVISION", os.environ.get("END_DATE", "latest"))
private   = os.environ.get("HF_PRIVATE", "true").lower() in ("1","true","yes")
token     = os.environ["HF_TOKEN"]

api = HfApi(token=token)
# Создать репозиторий, если его ещё нет
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

# Загрузить папку с моделями (веса, словари и т.п.)
api.upload_folder(
    folder_path="models",
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"nightly publish: {revision}",
    revision=revision
)

# Опционально: загрузить вспомогательные конфиги
if os.path.exists("configs/calibration.json"):
    api.upload_file(
        path_or_fileobj="configs/calibration.json",
        path_in_repo="configs/calibration.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"calibration: {revision}",
        revision=revision
)
print(f"Published to hf.co/{repo_id}@{revision}")
