import os, sys

if os.getenv("CI_DRY_RUN") == "1":
    print("CI_DRY_RUN=1 -> skip backtest heavy run")
    sys.exit(0)
print("backtest: implement backtest here")
