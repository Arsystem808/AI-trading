import argparse, json, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# Простейшая калибровка: подбираем смещение b в сигмоиде для M7/Octopus, остальное по умолчанию
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="config/calibration.json")
    args = ap.parse_args()

    try:
        base = json.load(open(args.out, "r", encoding="utf-8"))
    except Exception:
        base = {
            "Global":     {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
            "M7":         {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
            "W7":         {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
            "AlphaPulse": {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
            "Octopus":    {"conf":{"method":"sigmoid","params":{"a":1.2,"b":-0.10}}}
        }

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["atr14", "vol", "slope", "y"])
    # Накидываем суррогатную «сырую» уверенность как логит базового шанса для подбора b
    raw = 0.5 + 0.2*np.tanh((df["slope"].values)*500.0) - 0.1*np.clip(df["vol"].values-0.3, 0, 1)
    raw = np.clip(raw, 0.01, 0.99)
    y = df["y"].values.astype(int)
    # подберём b, который минимизирует брайер
    bs_best, b_best = 1e9, 0.0
    for b in np.linspace(-0.3, 0.3, 25):
        p = 1.0/(1.0+np.exp(-(1.0*raw + b)))
        bs = brier_score_loss(y, p)
        if bs < bs_best:
            bs_best, b_best = bs, b

    base["M7"]["conf"]["params"]["b"] = float(b_best)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
