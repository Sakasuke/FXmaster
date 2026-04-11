"""Training pipeline for the trend continuation model.

We use a chronological train/test split (NOT random!) because shuffling
financial time series leaks future information into the training set.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from fxmaster.ai.model import TrendModel
from fxmaster.config import ModelConfig
from fxmaster.features.engineer import FeatureBundle

logger = logging.getLogger("fxmaster.ai.trainer")


def _chronological_split(bundle: FeatureBundle, test_size: float):
    n = len(bundle.features)
    split_idx = int(n * (1 - test_size))
    X_train = bundle.features.iloc[:split_idx]
    y_train = bundle.label.iloc[:split_idx]
    X_test = bundle.features.iloc[split_idx:]
    y_test = bundle.label.iloc[split_idx:]
    return X_train, y_train, X_test, y_test


def train_model(
    bundle: FeatureBundle,
    cfg: ModelConfig,
    instrument: str,
) -> TrendModel:
    """Train a LightGBM model and return a serialisable TrendModel wrapper."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - lightgbm is a listed dep
        raise RuntimeError(
            "lightgbm is required for training. Install it via `pip install lightgbm`."
        ) from exc

    X_train, y_train, X_test, y_test = _chronological_split(bundle, cfg.test_size)
    logger.info(
        "Training set: %d rows (%.2f%% positive), Test set: %d rows (%.2f%% positive)",
        len(X_train),
        100 * float(y_train.mean()) if len(y_train) else 0.0,
        len(X_test),
        100 * float(y_test.mean()) if len(y_test) else 0.0,
    )

    train_ds = lgb.Dataset(X_train.to_numpy(), label=y_train.to_numpy())
    valid_ds = lgb.Dataset(X_test.to_numpy(), label=y_test.to_numpy(), reference=train_ds)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
    }
    params.update(cfg.params or {})

    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=400,
        valid_sets=[valid_ds],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    probas = np.asarray(booster.predict(X_test.to_numpy()))
    try:
        auc = float(roc_auc_score(y_test, probas))
    except ValueError:
        auc = float("nan")
    preds = (probas >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, preds)) if len(y_test) else float("nan")

    metrics = {
        "test_rows": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()) if len(y_test) else 0.0,
        "auc": auc,
        "accuracy": acc,
    }
    logger.info("Evaluation metrics: %s", metrics)

    return TrendModel(
        booster=booster,
        feature_names=list(X_train.columns),
        params=params,
        trained_on=datetime.now(timezone.utc).isoformat(),
        metrics=metrics,
    )
