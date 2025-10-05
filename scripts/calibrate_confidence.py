import os, sys, json, numpy as np
from pathlib import Path
if os.getenv("CI_DRY_RUN") == "1":
    print("CI_DRY_RUN=1 -> skip calibrate_confidence")
    sys.exit(0)

# простая калибровка: линейные коэффициенты по умолчанию
calib = {
  "Global":     {"a": 1.0, "b": 0.0},
  "M7":         {"a": 1.0, "b": 0.0},
  "W7":         {"a": 1.0, "b": 0.0},
  "AlphaPulse": {"a": 1.0, "b": 0.0},
  "Octopus":    {"a": 1.0, "b": 0.0}
}
Path("config").mkdir(exist_ok=True)
with open("config/calibration.json","w",encoding="utf-8") as f:
    json.dump(calib, f, ensure_ascii=False, indent=2)
print("saved config/calibration.json")
