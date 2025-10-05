import os, sys, json, numpy as np
from pathlib import Path
if os.getenv("CI_DRY_RUN") == "1":
    print("CI_DRY_RUN=1 -> skip train_octopus_meta")
    sys.exit(0)

# мета-модель пока заглушка: создаём файл-маркер
Path("models").mkdir(exist_ok=True)
open("models/octopus_meta.joblib","wb").write(b"placeholder")
print("saved models/octopus_meta.joblib")
