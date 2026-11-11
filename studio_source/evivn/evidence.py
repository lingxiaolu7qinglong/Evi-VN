from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import EvidenceConfig, HardRegionConfig
from .types import CrossFittedScores, EviVNInputs, Fold, validate_folds


EPSILON = 1e-5


class ProbabilityEstimator(Protocol):
    def fit(self, features: np.ndarray, labels: np.ndarray, **kwargs: object) -> object: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


EstimatorFactory = Callable[[int], ProbabilityEstimator]


def logistic_estimator(seed: int) -> ProbabilityEstimator:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def xgboost_estimator(seed: int) -> ProbabilityEstimator:
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=4,
        reg_lambda=4.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )


class ConstantEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, EPSILON, 1.0 - EPSILON))

    def fit(self, features: np.ndarray, labels: np.ndarray, **kwargs: object) -> "ConstantEstimator":
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), self.probability, dtype=np.float32)
        return np.column_stack((1.0 - positive, positive))


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        self.constant: float | None = None

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        if len(probability) == 0 or len(np.unique(labels)) < 2:
            self.constant = float(np.mean(labels)) if len(labels) else 0.5
        else:
            self.model.fit(self._logit(probability), labels)
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(probability), self.constant, dtype=np.float32)
        return self.model.predict_proba(self._logit(probability))[:, 1].astype(np.float32)


def _cross_fitted_calibration(
    raw_oof: np.ndarray,
    raw_full: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    folds: Sequence[Fold],
    eligible_oof: np.ndarray | None = None,
    eligible_full: np.ndarray | None = None,
    neutral: float | None = None,
) -> CrossFittedScores:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    oof_eligible = (
        np.ones(len(targets), dtype=bool)
        if eligible_oof is None
        else np.asarray(eligible_oof, dtype=bool)
    )
    full_eligible = (
        np.ones(len(targets), dtype=bool)
        if eligible_full is None
        else np.asarray(eligible_full, dtype=bool)
    )
    if len(oof_eligible) != len(targets) or len(full_eligible) != len(targets):
        raise ValueError("calibration eligibility masks must align with targets")

    oof = np.full(len(targets), np.nan, dtype=np.float32)
    if neutral is not None:
        oof[train_indices] = neutral
    for fold in folds:
        fit_indices = fold.train[oof_eligible[fold.train]]
        predict_indices = fold.valid[oof_eligible[fold.valid]]
        if len(predict_indices):
            calibrator = PlattCalibrator().fit(
                raw_oof[fit_indices], targets[fit_indices]
            )
            oof[predict_indices] = calibrator.transform(raw_oof[predict_indices])

    full = np.full(len(targets), neutral if neutral is not None else np.nan, dtype=np.float32)
    full_fit = train_indices[oof_eligible[train_indices]]
    full_calibrator = PlattCalibrator().fit(raw_oof[full_fit], targets[full_fit])
    predict_indices = np.flatnonzero(full_eligible)
    if len(predict_indices):
        full[predict_indices] = full_calibrator.transform(raw_full[predict_indices])
    return CrossFittedScores(oof, full, train_indices)


@dataclass(frozen=True)
class EvidenceBranch:
    hard_labels: np.ndarray
    router: CrossFittedScores
    prior: CrossFittedScores
    route_threshold: float
    route_oof: np.ndarray
    route_full: np.ndarray

    def split_aware_route(self) -> np.ndarray:
        route = np.asarray(self.route_full, dtype=bool).copy()
        route[self.router.train_indices] = self.route_oof[self.router.train_indices]
        return route


def consensus_hard_labels(
    discovery_oof: dict[str, np.ndarray] | np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    config: HardRegionConfig,
) -> np.ndarray:
    if isinstance(discovery_oof, dict):
        matrix = np.column_stack(list(discovery_oof.values()))
    else:
        matrix = np.asarray(discovery_oof)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("discovery_oof must have shape [num_nodes, num_backbones]")
    if not np.isfinite(matrix[train_indices]).all():
        raise ValueError("discovery OOF predictions are incomplete on training nodes")

    backbone_count = matrix.shape[1]
    error_votes = math.ceil(config.error_fraction * backbone_count)
    hard_votes = math.ceil(config.hard_fraction * backbone_count)
    prediction = matrix[train_indices] >= 0.5
    wrong = prediction != labels[train_indices, None]
    uncertain = np.abs(matrix[train_indices] - 0.5) <= config.uncertainty_delta
    hard_train = (wrong.sum(axis=1) >= error_votes) | (
        (wrong | uncertain).sum(axis=1) >= hard_votes
    )
    hard = np.zeros(len(labels), dtype=np.int64)
    hard[train_indices] = hard_train.astype(np.int64)
    return hard


def _balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = len(labels) / np.maximum(2.0 * counts, 1.0)
    return weights[labels]


def _fit_estimator(
    factory: EstimatorFactory,
    seed: int,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> ProbabilityEstimator:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or len(np.unique(labels[indices])) < 2:
        probability = float(np.mean(labels[indices])) if len(indices) else 0.5
        return ConstantEstimator(probability)
    model = factory(seed)
    sample_weight = _balanced_sample_weight(labels[indices])
    if isinstance(model, Pipeline):
        final_name = model.steps[-1][0]
        model.fit(
            features[indices],
            labels[indices],
            **{f"{final_name}__sample_weight": sample_weight},
        )
    else:
        model.fit(features[indices], labels[indices], sample_weight=sample_weight)
    return model


def _positive_probability(model: ProbabilityEstimator, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)


def fit_evidence_branch(
    inputs: EviVNInputs,
    hard_labels: np.ndarray,
    folds: Sequence[Fold],
    router_factory: EstimatorFactory,
    prior_factory: EstimatorFactory,
    config: EvidenceConfig,
    seed: int,
) -> EvidenceBranch:
    validate_folds(folds, inputs.split.train)
    evidence = np.asarray(inputs.evidence_features)
    labels = np.asarray(inputs.labels)
    train = np.asarray(inputs.split.train, dtype=np.int64)

    router_raw_oof = np.full(len(labels), np.nan, dtype=np.float32)
    for fold_id, fold in enumerate(folds):
        model = _fit_estimator(
            router_factory,
            seed + 100 + fold_id,
            evidence,
            hard_labels,
            fold.train,
        )
        router_raw_oof[fold.valid] = _positive_probability(model, evidence[fold.valid])
    router_full_model = _fit_estimator(
        router_factory, seed + 900, evidence, hard_labels, train
    )
    router_raw_full = _positive_probability(router_full_model, evidence)
    router_scores = _cross_fitted_calibration(
        router_raw_oof,
        router_raw_full,
        hard_labels,
        train,
        folds,
    )
    router_oof = router_scores.train_oof
    router_full = router_scores.full_train

    threshold = float(config.route_threshold)
    route_oof = np.zeros(len(labels), dtype=bool)
    route_oof[train] = router_oof[train] >= threshold
    route_full = router_full >= threshold

    prior_raw_oof = np.full(len(labels), np.nan, dtype=np.float32)
    prior_raw_oof[train] = 0.5
    for fold_id, fold in enumerate(folds):
        routed_train = fold.train[route_oof[fold.train]]
        model = _fit_estimator(
            prior_factory,
            seed + 1000 + fold_id,
            evidence,
            labels,
            routed_train,
        )
        routed_valid = fold.valid[route_oof[fold.valid]]
        if len(routed_valid):
            prior_raw_oof[routed_valid] = _positive_probability(
                model, evidence[routed_valid]
            )

    routed_train = train[route_oof[train]]
    prior_full_model = _fit_estimator(
        prior_factory, seed + 1900, evidence, labels, routed_train
    )
    prior_raw_full = np.full(len(labels), 0.5, dtype=np.float32)
    routed_nodes = np.flatnonzero(route_full)
    if len(routed_nodes):
        prior_raw_full[routed_nodes] = _positive_probability(
            prior_full_model, evidence[routed_nodes]
        )

    prior_scores = _cross_fitted_calibration(
        prior_raw_oof,
        prior_raw_full,
        labels,
        train,
        folds,
        eligible_oof=route_oof,
        eligible_full=route_full,
        neutral=0.5,
    )

    return EvidenceBranch(
        hard_labels=np.asarray(hard_labels, dtype=np.int64),
        router=router_scores,
        prior=prior_scores,
        route_threshold=threshold,
        route_oof=route_oof,
        route_full=route_full,
    )
