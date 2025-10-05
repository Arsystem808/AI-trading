from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["MeanReversionSignalEngine"]


class MeanReversionSignalEngine:
    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        ma_fast: int = 20,
        ma_slow: int = 50,
        z_period: int = 20,
        z_long_thr: float = -1.8,
        z_short_thr: float = 1.8,
        rsi_os: int = 30,
        rsi_ob: int = 70,
        use_adaptive_rsi: bool = True,
        atr_period: int = 14,
        sr_lookback: int = 20,
        near_sr_k_atr: float = 0.5,
        volume_ratio_min: float = 1.2,
        bb_period: int = 20,
        bb_std: float = 2.0,
        require_range_regime: bool = True,
        range_thr_pct: float = 2.0,
        confidence_min: float = 60.0,
        tp_atr_mult: float = 1.5,
        sl_atr_mult: float = 2.0,
        hold_time_bars: int = 20,
        cooldown_bars: int = 10,
        min_avg_volume: Optional[float] = 2e5,
        warmup_bars: int = 200,
    ):
        (
            self.rsi_period,
            self.stoch_period,
            self.ma_fast,
            self.ma_slow,
            self.z_period,
        ) = (
            rsi_period,
            stoch_period,
            ma_fast,
            ma_slow,
            z_period,
        )
        (
            self.z_long_thr,
            self.z_short_thr,
            self.rsi_os,
            self.rsi_ob,
            self.use_adaptive_rsi,
        ) = (
            z_long_thr,
            z_short_thr,
            rsi_os,
            rsi_ob,
            use_adaptive_rsi,
        )
        self.atr_period, self.sr_lookback, self.near_sr_k_atr, self.volume_ratio_min = (
            atr_period,
            sr_lookback,
            near_sr_k_atr,
            volume_ratio_min,
        )
        self.bb_period, self.bb_std, self.require_range_regime, self.range_thr_pct = (
            bb_period,
            bb_std,
            require_range_regime,
            range_thr_pct,
        )
        self.confidence_min, self.tp_atr_mult, self.sl_atr_mult = (
            confidence_min,
            tp_atr_mult,
            sl_atr_mult,
        )
        (
            self.hold_time_bars,
            self.cooldown_bars,
            self.min_avg_volume,
            self.warmup_bars,
        ) = (
            hold_time_bars,
            cooldown_bars,
            min_avg_volume,
            warmup_bars,
        )
        self._cooldowns: Dict[str, int] = {}
        self.filter_weights = {
            "base_mr": 1.6,
            "volume": 1.1,
            "sr": 1.2,
            "vol_ok": 1.0,
            "bb_extreme": 1.1,
            "stoch_extreme": 1.2,
            "momentum_turn": 1.2,
            "regime_range": 1.2,
        }

    @staticmethod
    def _rsi(c: pd.Series, n: int) -> pd.Series:
        d = c.diff()
        up = d.clip(lower=0)
        dn = (-d).clip(lower=0)
        ru = up.ewm(alpha=1 / n, adjust=False).mean()
        rd = dn.ewm(alpha=1 / n, adjust=False).mean()
        rs = ru / rd.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, n: int) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / n, adjust=False).mean()

    @staticmethod
    def _macd(c: pd.Series) -> pd.Series:
        return c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()

    @staticmethod
    def _stoch(df: pd.DataFrame, n: int) -> pd.Series:
        ll = df["low"].rolling(n).min()
        hh = df["high"].rolling(n).max()
        rng = (hh - ll).replace(0, np.nan)
        return 100 * (df["close"] - ll) / rng

    @staticmethod
    def _safe(a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, np.nan)

    def _feats(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df.copy()
        o["MAF"] = o["close"].rolling(self.ma_fast, min_periods=self.ma_fast).mean()
        o["MAS"] = o["close"].rolling(self.ma_slow, min_periods=self.ma_slow).mean()
        mu = o["close"].rolling(self.z_period, min_periods=self.z_period).mean()
        sd = o["close"].rolling(self.z_period, min_periods=self.z_period).std(ddof=0)
        o["Z"] = self._safe(o["close"] - mu, sd)
        o["RSI"] = self._rsi(o["close"], self.rsi_period)
        o["ATR"] = self._atr(o, self.atr_period)
        o["VolPct"] = self._safe(o["ATR"], o["close"]) * 100
        o["VolRank"] = o["VolPct"].rolling(60, min_periods=25).rank(pct=True)
        o["VolMA"] = o["volume"].rolling(20, min_periods=10).mean()
        o["VolRatio"] = self._safe(o["volume"], o["VolMA"])
        o["Res"] = o["high"].rolling(20, min_periods=20).max()
        o["Sup"] = o["low"].rolling(20, min_periods=20).min()
        dS = (o["close"] - o["Sup"]).abs()
        dR = (o["close"] - o["Res"]).abs()
        thr = self.near_sr_k_atr * o["ATR"]
        o["NearSup"] = dS <= thr
        o["NearRes"] = dR <= thr
        mid = o["close"].rolling(20, min_periods=20).mean()
        std = o["close"].rolling(20, min_periods=20).std(ddof=0)
        o["BB_Up"] = mid + 2 * std
        o["BB_Lo"] = mid - 2 * std
        width = (o["BB_Up"] - o["BB_Lo"]).replace(0, np.nan)
        o["BB_Pos"] = (o["close"] - o["BB_Lo"]) / width
        o["Stoch"] = self._stoch(o, self.stoch_period)
        o["MACD"] = self._macd(o["close"])
        ms = o["MAS"].replace(0, np.nan)
        o["TrendStrength"] = (o["MAF"] - o["MAS"]).abs() / ms * 100
        o["Regime"] = np.where(
            o["TrendStrength"] <= self.range_thr_pct, "Range", "Trend"
        )
        return o

    def _rsi_bounds(self, vol_ann: float) -> tuple[int, int]:
        return (
            (35, 65)
            if (self.use_adaptive_rsi and vol_ann > 0.40)
            else (self.rsi_os, self.rsi_ob)
        )

    def clear_state(self) -> None:
        self._cooldowns.clear()

    def set_cooldown(self, s: str, b: int) -> None:
        self._cooldowns[s] = max(0, int(b))

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        req = {"open", "high", "low", "close", "volume"}
        if not req.issubset(df.columns):
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "OPEN": "open",
                    "HIGH": "high",
                    "LOW": "low",
                    "CLOSE": "close",
                    "VOLUME": "volume",
                }
            )
        if not req.issubset(df.columns):
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "timestamp",
                    "side",
                    "price",
                    "confidence",
                    "reasons",
                    "atr",
                    "sl_distance",
                    "tp_distance",
                    "order_hint",
                    "hold_time_bars",
                    "cooldown_bars",
                    "z",
                    "rsi",
                    "stoch",
                    "bb_pos",
                    "vol_rank",
                ]
            )
        df = df.copy()
        vol_ma30 = df["volume"].rolling(30, min_periods=10).mean()
        if (
            vol_ma30.empty
            or pd.isna(vol_ma30.iloc[-1])
            or (self.min_avg_volume and vol_ma30.iloc[-1] < self.min_avg_volume)
        ):
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "timestamp",
                    "side",
                    "price",
                    "confidence",
                    "reasons",
                    "atr",
                    "sl_distance",
                    "tp_distance",
                    "order_hint",
                    "hold_time_bars",
                    "cooldown_bars",
                    "z",
                    "rsi",
                    "stoch",
                    "bb_pos",
                    "vol_rank",
                ]
            )
        f = self._feats(df)
        r = f["close"].pct_change().dropna()
        vol = float(r.std() * np.sqrt(252)) if len(r) else 0.30
        rsi_os, rsi_ob = self._rsi_bounds(vol)
        start = min(
            max(self.warmup_bars, self.ma_slow + 5, self.atr_period + 5, 25),
            max(0, len(f) - 1),
        )
        cd = int(self._cooldowns.get(symbol, 0))
        sigs: List[Dict] = []

        def add(sc: Dict[str, float], name: str, ok: bool) -> None:
            w = self.filter_weights[name]
            sc["max"] += w
            if ok:
                sc["sum"] += w
                sc["reasons"].append(name)

        for i in range(start, len(f)):
            row = f.iloc[i]
            prev = f.iloc[i - 1] if i > 0 else row
            if cd > 0:
                cd -= 1
                continue
            base_long = (pd.notna(row["Z"]) and row["Z"] <= self.z_long_thr) and (
                pd.notna(row["RSI"]) and row["RSI"] <= rsi_os
            )
            base_short = (pd.notna(row["Z"]) and row["Z"] >= self.z_short_thr) and (
                pd.notna(row["RSI"]) and row["RSI"] >= rsi_ob
            )
            if not (base_long or base_short):
                continue
            sc = {"sum": 0.0, "max": 0.0, "reasons": []}
            add(sc, "base_mr", True)
            add(
                sc,
                "volume",
                pd.notna(row["VolRatio"]) and row["VolRatio"] >= self.volume_ratio_min,
            )
            add(sc, "sr", bool(row["NearSup"]) if base_long else bool(row["NearRes"]))
            add(
                sc, "vol_ok", pd.notna(row["VolRank"]) and float(row["VolRank"]) >= 0.30
            )
            add(
                sc,
                "bb_extreme",
                pd.notna(row["BB_Pos"])
                and ((row["BB_Pos"] <= 0.10) if base_long else (row["BB_Pos"] >= 0.90)),
            )
            add(
                sc,
                "stoch_extreme",
                pd.notna(row["Stoch"])
                and ((row["Stoch"] <= 20.0) if base_long else (row["Stoch"] >= 80.0)),
            )
            add(
                sc,
                "momentum_turn",
                pd.notna(row["MACD"])
                and pd.notna(prev["MACD"])
                and (
                    (row["MACD"] > prev["MACD"])
                    if base_long
                    else (row["MACD"] < prev["MACD"])
                ),
            )
            add(
                sc,
                "regime_range",
                (row["Regime"] == "Range") if self.require_range_regime else True,
            )
            conf = (sc["sum"] / sc["max"] * 100.0) if sc["max"] > 0 else 0.0
            if conf < self.confidence_min:
                continue
            side = "LONG" if base_long else "SHORT"
            price = float(row["close"])
            atr = float(row["ATR"]) if pd.notna(row["ATR"]) else float(np.nan)
            sl_d = float(self.sl_atr_mult * atr) if np.isfinite(atr) else float(np.nan)
            tp_d = float(self.tp_atr_mult * atr) if np.isfinite(atr) else float(np.nan)
            order = "buy-limit" if side == "LONG" else "sell-limit"
            sigs.append(
                {
                    "symbol": symbol,
                    "timestamp": f.index[i],
                    "side": side,
                    "price": price,
                    "confidence": round(float(conf), 2),
                    "reasons": ", ".join(sc["reasons"]),
                    "atr": atr,
                    "sl_distance": sl_d,
                    "tp_distance": tp_d,
                    "order_hint": order,
                    "hold_time_bars": int(self.hold_time_bars),
                    "cooldown_bars": int(self.cooldown_bars),
                    "z": float(row["Z"]) if pd.notna(row["Z"]) else float(np.nan),
                    "rsi": float(row["RSI"]) if pd.notna(row["RSI"]) else float(np.nan),
                    "stoch": (
                        float(row["Stoch"]) if pd.notna(row["Stoch"]) else float(np.nan)
                    ),
                    "bb_pos": (
                        float(row["BB_Pos"])
                        if pd.notna(row["BB_Pos"])
                        else float(np.nan)
                    ),
                    "vol_rank": (
                        float(row["VolRank"])
                        if pd.notna(row["VolRank"])
                        else float(np.nan)
                    ),
                }
            )
            cd = int(self.cooldown_bars)
        self._cooldowns[symbol] = cd
        cols = [
            "symbol",
            "timestamp",
            "side",
            "price",
            "confidence",
            "reasons",
            "atr",
            "sl_distance",
            "tp_distance",
            "order_hint",
            "hold_time_bars",
            "cooldown_bars",
            "z",
            "rsi",
            "stoch",
            "bb_pos",
            "vol_rank",
        ]
        return pd.DataFrame(sigs, columns=cols)
