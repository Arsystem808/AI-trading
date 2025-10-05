import os, sys
if os.getenv("CI_DRY_RUN") == "1":
    print("CI_DRY_RUN=1 -> skip snapshots heavy run")
    sys.exit(0)
print("snapshots: implement snapshots here")
