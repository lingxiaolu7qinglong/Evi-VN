from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from .config import FinalizerConfig
from .types import CrossFittedScores, Split


EPSILON = 1e-5


def clipped_logit(probability: np.ndarray, bound: float = 8.0) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.clip(np.log(clipped / (1.0 - clipped)), -bound, bound)


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def finalizer_features(
    augmented_probability: np.ndarray,
    calibrated_prior: np.ndarray,
    graph_probability: np.ndarray,
    router_score: np.ndarray,
    route_mask: np.ndarray,
) -> np.ndarray:
    """Return [augmented logit, prior logit, graph logit, router, route]."""
    return np.column_stack(
        (
            logit(augmented_probability),
            clipped_logit(calibrated_prior),
            logit(graph_probability),
            np.asarray(router_score, dtype=np.float32),
            np.asarray(route_mask, dtype=np.float32),
        )
    ).astype(np.float32)


@dataclass(frozen=True)
class FinalizerMatrices:
    train: np.ndarray
    dev: np.ndarray
    test: np.ndarray


def split_aware_finalizer_matrices(
    split: Split,
    graph: CrossFittedScores,
    augmented: CrossFittedScores,
    prior: CrossFittedScores,
    router: CrossFittedScores,
    route_oof: np.ndarray,
    route_full: np.ndarray,
) -> FinalizerMatrices:
    train = finalizer_features(
        augmented.train_oof[split.train],
        prior.train_oof[split.train],
        graph.train_oof[split.train],
        router.train_oof[split.train],
        route_oof[split.train],
    )
    dev = finalizer_features(
        augmented.full_train[split.dev],
        prior.full_train[split.dev],
        graph.full_train[split.dev],
        router.full_train[split.dev],
        route_full[split.dev],
    )
    test = finalizer_features(
        augmented.full_train[split.test],
        prior.full_train[split.test],
        graph.full_train[split.test],
        router.full_train[split.test],
        route_full[split.test],
    )
    if not all(np.isfinite(matrix).all() for matrix in (train, dev, test)):
        raise ValueError("finalizer received an incomplete split-aware score")
    return FinalizerMatrices(train, dev, test)


def binary_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = probability >= threshold
    auc = float(roc_auc_score(labels, probability)) if len(np.unique(labels)) > 1 else 0.5
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "auc": auc,
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "threshold": float(threshold),
    }


def _select_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    best: tuple[float, float, float, float] | None = None
    best_threshold = 0.5
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = binary_metrics(labels, probability, float(threshold))
        composite = (metrics["accuracy"] + metrics["auc"] + metrics["f1"]) / 3.0
        candidate = (
            composite,
            metrics["f1"],
            metrics["accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    return best_threshold


@dataclass(frozen=True)
class FittedFinalizer:
    model: object
    threshold: float
    development_metrics: dict[str, float]
    model_name: str


def _sample_weight(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = len(labels) / np.maximum(2.0 * counts, 1.0)
    return weights[labels]


def fit_score_finalizer(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    dev_labels: np.ndarray,
    config: FinalizerConfig,
    seed: int,
) -> FittedFinalizer:
    candidates: list[tuple[str, object]] = []
    for regularization in config.logistic_c:
        candidates.append(
            (
                f"logistic_C={regularization:g}",
                LogisticRegression(
                    C=regularization,
                    max_iter=3000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        )
    if config.include_xgboost:
        for depth, estimators in ((1, 100), (2, 160)):
            candidates.append(
                (
                    f"xgboost_depth={depth}",
                    XGBClassifier(
                        n_estimators=estimators,
                        max_depth=depth,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.80,
                        min_child_weight=4,
                        reg_lambda=4.0,
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=4,
                    ),
                )
            )

    selected: tuple[tuple[float, float, float], FittedFinalizer] | None = None
    for name, model in candidates:
        if isinstance(model, XGBClassifier):
            model.fit(
                train_features,
                train_labels,
                sample_weight=_sample_weight(train_labels),
            )
        else:
            model.fit(train_features, train_labels)
        probability = model.predict_proba(dev_features)[:, 1]
        threshold = (
            _select_threshold(dev_labels, probability)
            if config.tune_decision_threshold
            else 0.5
        )
        metrics = binary_metrics(dev_labels, probability, threshold)
        key = (
            (metrics["accuracy"] + metrics["auc"] + metrics["f1"]) / 3.0,
            metrics["auc"],
            metrics["f1"],
        )
        fitted = FittedFinalizer(model, threshold, metrics, name)
        if selected is None or key > selected[0]:
            selected = (key, fitted)
    if selected is None:
        raise RuntimeError("no finalizer candidate was fitted")
    return selected[1]
