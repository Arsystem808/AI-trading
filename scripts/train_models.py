import os, sys
if os.getenv("CI_DRY_RUN") == "1":
    print("CI_DRY_RUN=1 -> skip train_models heavy run")
    sys.exit(0)
print("train_models: implement training here")
