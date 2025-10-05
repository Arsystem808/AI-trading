import os
import sys
from pathlib import Path

if os.getenv("CI_DRY_RUN") == "1":
    print("CI_DRY_RUN=1 -> skip train_octopus_meta")
    sys.exit(0)

Path("models").mkdir(exist_ok=True)
Path("models/octopus_meta.joblib").write_bytes(b"placeholder")
print("saved models/octopus_meta.joblib")
