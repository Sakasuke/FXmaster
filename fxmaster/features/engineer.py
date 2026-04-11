"""Multi-timeframe feature engineering.

Builds a feature DataFrame from LTF and HTF candles by:
  1. Computing indicators on each timeframe
  2. Forward-filling HTF features onto the LTF timeline (to avoid look-ahead)
  3. Encoding Dow-theory-style swing structure (higher highs / lower lows)
  4. Producing a label column for supervised training

The output DataFrame is what the AI engine consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxmaster.config import ModelConfig, StrategyConfig
from fxmaster.features.indicators import adx, atr, bollinger_bands, ema, macd, rsi
from fxmaster.utils.pip_math import price_to_pips


@dataclass
class FeatureBundle:
    """Container for features, label, and the aligned close price."""

    features: pd.DataFrame
    label: pd.Series
    close: pd.Series
    time: pd.Series


def _compute_indicators(df: pd.DataFrame, strat: StrategyConfig, prefix: str = "") -> pd.DataFrame:
    """Compute indicator columns in-place on a candle DataFrame."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    out[f"{prefix}ema_fast"] = ema(close, strat.ema_fast)
    out[f"{prefix}ema_slow"] = ema(close, strat.ema_slow)
    out[f"{prefix}ema_diff"] = out[f"{prefix}ema_fast"] - out[f"{prefix}ema_slow"]
    out[f"{prefix}rsi"] = rsi(close, strat.rsi_period)
    out[f"{prefix}atr"] = atr(high, low, close, strat.atr_period)
    out[f"{prefix}adx"] = adx(high, low, close, strat.atr_period)

    macd_line, signal_line, hist = macd(close)
    out[f"{prefix}macd"] = macd_line
    out[f"{prefix}macd_signal"] = signal_line
    out[f"{prefix}macd_hist"] = hist

    upper, mid, lower = bollinger_bands(close)
    width = (upper - lower) / mid.replace(0, np.nan)
    out[f"{prefix}bb_width"] = width
    out[f"{prefix}bb_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)

    return out


def _swing_structure(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Numerically encode Dow-theory swing structure.

    ``swing_trend`` = +1 when recent highs and lows are both rising,
                       -1 when both are falling,
                       0 otherwise.
    """
    high = df["high"]
    low = df["low"]

    rolling_high = high.rolling(window=window, min_periods=window).max()
    rolling_low = low.rolling(window=window, min_periods=window).min()

    hh = rolling_high.diff() > 0
    hl = rolling_low.diff() > 0
    lh = rolling_high.diff() < 0
    ll = rolling_low.diff() < 0

    trend = pd.Series(0, index=df.index, dtype="int8")
    trend[hh & hl] = 1
    trend[lh & ll] = -1

    out = pd.DataFrame(index=df.index)
    out["swing_trend"] = trend
    out["rolling_high"] = rolling_high
    out["rolling_low"] = rolling_low
    out["pullback_depth"] = (rolling_high - df["close"]) / (rolling_high - rolling_low).replace(0, np.nan)
    return out


def _align_htf_to_ltf(ltf: pd.DataFrame, htf_features: pd.DataFrame, htf_time: pd.Series) -> pd.DataFrame:
    """Forward-fill HTF features onto the LTF timeline without leaking the future.

    For each LTF bar at time T we pick the latest HTF bar whose close time is <= T.
    """
    htf_indexed = htf_features.copy()
    htf_indexed["time"] = pd.to_datetime(htf_time.values, utc=True)
    htf_indexed = htf_indexed.sort_values("time").reset_index(drop=True)

    ltf_times = pd.to_datetime(ltf["time"].values, utc=True)
    ltf_df = pd.DataFrame({"time": ltf_times})
    merged = pd.merge_asof(
        ltf_df,
        htf_indexed,
        on="time",
        direction="backward",
    )
    return merged.drop(columns=["time"])


def _make_label(
    ltf: pd.DataFrame,
    instrument: str,
    horizon_bars: int,
    threshold_pips: float,
) -> pd.Series:
    """Binary label: 1 if the close moves > threshold_pips in our favor within ``horizon_bars``.

    We use a directional-agnostic definition focused on "did a trend move happen at all",
    because the strategy layer decides direction from the HTF trend filter.
    """
    close = ltf["close"]
    future_high = ltf["high"].shift(-1).rolling(window=horizon_bars, min_periods=1).max().shift(1 - horizon_bars)
    future_low = ltf["low"].shift(-1).rolling(window=horizon_bars, min_periods=1).min().shift(1 - horizon_bars)

    up_move_pips = price_to_pips((future_high - close).fillna(0), instrument)
    down_move_pips = price_to_pips((close - future_low).fillna(0), instrument)

    moved = (up_move_pips >= threshold_pips) | (down_move_pips >= threshold_pips)
    return moved.astype("int8")


def build_features(
    ltf: pd.DataFrame,
    htf: pd.DataFrame,
    instrument: str,
    strategy_cfg: StrategyConfig,
    model_cfg: ModelConfig | None = None,
) -> FeatureBundle:
    """Build the feature bundle used for training and live prediction.

    Parameters
    ----------
    ltf:
        Lower timeframe candles (e.g. 5-minute).
    htf:
        Higher timeframe candles (e.g. 1-hour).
    instrument:
        OANDA instrument name, used for pip conversion.
    strategy_cfg:
        Strategy parameters (indicator periods).
    model_cfg:
        Model parameters (label horizon / threshold). If ``None`` the
        label column will be all NaN (useful for live inference).
    """
    ltf = ltf.reset_index(drop=True).copy()
    htf = htf.reset_index(drop=True).copy()

    ltf_feats = _compute_indicators(ltf, strategy_cfg, prefix="ltf_")
    htf_feats = _compute_indicators(htf, strategy_cfg, prefix="htf_")
    swing = _swing_structure(ltf)

    htf_aligned = _align_htf_to_ltf(ltf, htf_feats, htf["time"])

    features = pd.concat([ltf_feats, htf_aligned, swing], axis=1)
    # Derived structural flags (what the strategy layer will want)
    features["htf_uptrend"] = (htf_aligned["htf_ema_diff"] > 0).astype("int8")
    features["htf_strong_trend"] = (htf_aligned["htf_adx"] >= strategy_cfg.adx_min).astype("int8")
    features["ltf_pullback_long"] = (
        (features["htf_uptrend"] == 1) & (ltf_feats["ltf_rsi"] <= strategy_cfg.rsi_pullback_long)
    ).astype("int8")
    features["ltf_pullback_short"] = (
        (features["htf_uptrend"] == 0) & (ltf_feats["ltf_rsi"] >= strategy_cfg.rsi_pullback_short)
    ).astype("int8")

    if model_cfg is not None:
        label = _make_label(
            ltf,
            instrument,
            horizon_bars=model_cfg.label_horizon_bars,
            threshold_pips=model_cfg.label_threshold_pips,
        )
    else:
        label = pd.Series(np.nan, index=ltf.index, dtype="float")

    bundle = FeatureBundle(
        features=features,
        label=label,
        close=ltf["close"].copy(),
        time=pd.to_datetime(ltf["time"], utc=True),
    )
    return bundle


def drop_na_rows(bundle: FeatureBundle) -> FeatureBundle:
    """Remove rows with NaNs in features or label (training path only)."""
    combined = pd.concat(
        [bundle.features, bundle.label.rename("__label__"), bundle.close.rename("__close__")],
        axis=1,
    )
    combined = combined.dropna().reset_index(drop=True)
    feature_cols = [c for c in combined.columns if c not in ("__label__", "__close__")]
    return FeatureBundle(
        features=combined[feature_cols],
        label=combined["__label__"].astype("int8"),
        close=combined["__close__"],
        time=pd.to_datetime(bundle.time.iloc[-len(combined):].reset_index(drop=True), utc=True),
    )
