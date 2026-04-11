"""Unit tests for the pure-pandas technical indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxmaster.features.indicators import adx, atr, bollinger_bands, ema, macd, rsi


def _synthetic_ohlc(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 5, n)
    noise = rng.normal(0, 0.3, n).cumsum()
    close = 100 + drift + noise
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    open_ = close + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1}
    )


def test_ema_final_value_reasonable():
    df = _synthetic_ohlc()
    result = ema(df["close"], 20)
    assert result.isna().sum() >= 19  # first period has NaN
    assert not np.isnan(result.iloc[-1])
    # EMA should be close to the last close for a gently drifting series.
    assert abs(result.iloc[-1] - df["close"].iloc[-1]) < 3.0


def test_rsi_within_0_100():
    df = _synthetic_ohlc()
    result = rsi(df["close"], 14)
    assert result.between(0, 100).all()


def test_atr_positive():
    df = _synthetic_ohlc()
    result = atr(df["high"], df["low"], df["close"], 14).dropna()
    assert (result > 0).all()


def test_macd_shapes():
    df = _synthetic_ohlc()
    m, s, h = macd(df["close"])
    assert len(m) == len(df)
    # Histogram = macd - signal
    diff = (m - s) - h
    np.testing.assert_allclose(diff.dropna().to_numpy(), 0.0, atol=1e-10)


def test_adx_in_reasonable_range():
    df = _synthetic_ohlc()
    a = adx(df["high"], df["low"], df["close"], 14).dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_bollinger_bands_ordering():
    df = _synthetic_ohlc()
    upper, middle, lower = bollinger_bands(df["close"], 20, 2)
    valid = pd.concat([upper, middle, lower], axis=1).dropna()
    assert (valid.iloc[:, 0] >= valid.iloc[:, 1]).all()
    assert (valid.iloc[:, 1] >= valid.iloc[:, 2]).all()
