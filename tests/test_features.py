"""Tests for feature engineering and the backtest engine wiring."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fxmaster.backtest import BacktestEngine
from fxmaster.config import ModelConfig, RiskConfig, StrategyConfig
from fxmaster.features.engineer import build_features, drop_na_rows
from fxmaster.strategy import TrendScalpStrategy


def _make_candles(n: int, granularity_min: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=i * granularity_min) for i in range(n)]
    drift = np.linspace(0, 4, n)
    noise = rng.normal(0, 0.05, n).cumsum()
    close = 150.0 + drift + noise
    high = close + rng.uniform(0.02, 0.08, n)
    low = close - rng.uniform(0.02, 0.08, n)
    open_ = close + rng.normal(0, 0.02, n)
    return pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n, dtype=int) * 100,
        }
    )


def test_build_features_shapes_and_label():
    ltf = _make_candles(3000, 5, seed=1)
    htf = _make_candles(400, 60, seed=2)
    strat = StrategyConfig()
    model_cfg = ModelConfig()
    bundle = build_features(ltf, htf, "USD_JPY", strat, model_cfg)
    assert len(bundle.features) == len(ltf)
    assert "ltf_rsi" in bundle.features.columns
    assert "htf_ema_fast" in bundle.features.columns
    assert "htf_uptrend" in bundle.features.columns
    assert bundle.label.dtype.kind in ("i", "u")


def test_drop_na_rows_produces_clean_matrix():
    ltf = _make_candles(3000, 5, seed=3)
    htf = _make_candles(400, 60, seed=4)
    bundle = build_features(ltf, htf, "USD_JPY", StrategyConfig(), ModelConfig())
    cleaned = drop_na_rows(bundle)
    assert not cleaned.features.isna().any().any()
    assert len(cleaned.features) > 0


def test_backtest_engine_runs_without_model():
    ltf = _make_candles(3000, 5, seed=5)
    htf = _make_candles(400, 60, seed=6)
    bundle = build_features(ltf, htf, "USD_JPY", StrategyConfig(), ModelConfig())
    # Keep alignment: drop rows where features have NaN
    mask = bundle.features.notna().all(axis=1)
    ltf_clean = ltf.loc[mask].reset_index(drop=True)
    bundle.features = bundle.features.loc[mask].reset_index(drop=True)
    bundle.label = bundle.label.loc[mask].reset_index(drop=True)
    bundle.close = bundle.close.loc[mask].reset_index(drop=True)

    strategy = TrendScalpStrategy(StrategyConfig(), model=None)
    engine = BacktestEngine(
        strategy=strategy,
        strategy_cfg=StrategyConfig(),
        risk_cfg=RiskConfig(),
        instrument="USD_JPY",
        starting_equity=1_000_000.0,
        spread_pips=0.8,
    )
    result = engine.run(ltf_clean, bundle)
    summary = result.summary()
    # The engine should execute, produce a summary dict, and not crash.
    assert "trades" in summary
    assert summary["final_equity"] >= 0
