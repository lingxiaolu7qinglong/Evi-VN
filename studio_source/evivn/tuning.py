from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .config import (
    DISCOVERY_BACKBONES,
    EviVNConfig,
    FinalizerConfig,
    ProxyRegConfig,
    VirtualNodeConfig,
)
from .evidence import (
    EstimatorFactory,
    EvidenceBranch,
    consensus_hard_labels,
    fit_evidence_branch,
    xgboost_estimator,
)
from .finalizer import (
    FinalizerMatrices,
    FittedFinalizer,
    binary_metrics,
    fit_score_finalizer,
    split_aware_finalizer_matrices,
)
from .pipeline import EviVNResult
from .training import FittedNodeModel, cross_fit_gnn, discovery_oof_predictions
from .types import CrossFittedScores, EviVNInputs, Fold, validate_folds


@dataclass(frozen=True)
class EstimatorCandidate:
    name: str
    factory: EstimatorFactory


def _default_virtual_candidates() -> tuple[VirtualNodeConfig, ...]:
    proxy = ProxyRegConfig()
    return (
        VirtualNodeConfig(0.40, 0.60, 1.0, proxy=proxy),
        VirtualNodeConfig(0.40, 0.60, 4.0, proxy=proxy),
        VirtualNodeConfig(0.45, 0.55, 2.0, proxy=proxy),
        VirtualNodeConfig(0.45, 0.55, 4.0, proxy=proxy),
    )


@dataclass(frozen=True)
class TuningSpace:
    router_candidates: tuple[EstimatorCandidate, ...] = field(
        default_factory=lambda: (EstimatorCandidate("xgboost", xgboost_estimator),)
    )
    prior_candidates: tuple[EstimatorCandidate, ...] = field(
        default_factory=lambda: (EstimatorCandidate("xgboost", xgboost_estimator),)
    )
    route_thresholds: tuple[float, ...] = tuple(
        step / 100 for step in range(1, 100)
    )
    virtual_node_candidates: tuple[VirtualNodeConfig, ...] = field(
        default_factory=_default_virtual_candidates
    )
    finalizer_candidates: tuple[FinalizerConfig, ...] = field(
        default_factory=lambda: (FinalizerConfig(),)
    )

    def __post_init__(self) -> None:
        if not self.router_candidates or not self.prior_candidates:
            raise ValueError("router and prior candidate sets cannot be empty")
        if not self.route_thresholds or not self.virtual_node_candidates:
            raise ValueError("route and virtual-node candidate sets cannot be empty")
        if not self.finalizer_candidates:
            raise ValueError("finalizer candidate set cannot be empty")
        if any(not 0.0 <= threshold <= 1.0 for threshold in self.route_thresholds):
            raise ValueError("route thresholds must be in [0, 1]")


@dataclass(frozen=True)
class TuningTrial:
    router_name: str
    prior_name: str
    route_threshold: float
    train_oof_route_coverage: float
    development_route_coverage: float
    virtual_node: VirtualNodeConfig
    finalizer: FinalizerConfig
    selected_finalizer: str
    decision_threshold: float
    development_metrics: dict[str, float]
    selection_score: float


@dataclass(frozen=True)
class EviVNTuningResult:
    best_result: EviVNResult
    best_trial: TuningTrial
    trials: tuple[TuningTrial, ...]


@dataclass(frozen=True)
class _BestPayload:
    evidence_branch: EvidenceBranch
    augmented_scores: CrossFittedScores
    augmented_model: FittedNodeModel
    finalizer_matrices: FinalizerMatrices
    finalizer: FittedFinalizer
    development_probability: np.ndarray


def _selection_key(
    metrics: dict[str, float],
) -> tuple[float, float, float]:
    accuracy = metrics["accuracy"]
    auc = metrics["auc"]
    f1 = metrics["f1"]
    return accuracy, auc, f1


class EviVNTuner:
    """Select the main Evi-VN configuration by final development accuracy."""

    selection_metric = "accuracy"

    def __init__(
        self,
        base_config: EviVNConfig | None = None,
        tuning_space: TuningSpace | None = None,
    ) -> None:
        self.base_config = EviVNConfig() if base_config is None else base_config
        self.tuning_space = TuningSpace() if tuning_space is None else tuning_space

    def tune(
        self,
        inputs: EviVNInputs,
        folds: Sequence[Fold],
        target_backbone: str,
        discovery_backbones: Sequence[str] = DISCOVERY_BACKBONES,
        discovery_oof: dict[str, np.ndarray] | None = None,
    ) -> EviVNTuningResult:
        inputs.validate()
        validate_folds(folds, inputs.split.train)
        target_backbone = target_backbone.lower()

        if discovery_oof is None:
            discovery_oof = discovery_oof_predictions(
                inputs,
                discovery_backbones,
                folds,
                self.base_config.gnn,
                self.base_config.seed,
            )
        hard_labels = consensus_hard_labels(
            discovery_oof,
            inputs.labels,
            inputs.split.train,
            self.base_config.hard_region,
        )
        graph_scores, graph_model = cross_fit_gnn(
            inputs,
            target_backbone,
            folds,
            self.base_config.gnn,
            self.base_config.seed + 30000,
            precomputed_oof=discovery_oof.get(target_backbone),
        )

        trials: list[TuningTrial] = []
        best_key: tuple[float, float, float] | None = None
        best_trial: TuningTrial | None = None
        best_payload: _BestPayload | None = None

        for router_candidate in self.tuning_space.router_candidates:
            for prior_candidate in self.tuning_space.prior_candidates:
                for route_threshold in self.tuning_space.route_thresholds:
                    evidence_config = replace(
                        self.base_config.evidence,
                        route_threshold=float(route_threshold),
                    )
                    branch = fit_evidence_branch(
                        inputs,
                        hard_labels,
                        folds,
                        router_candidate.factory,
                        prior_candidate.factory,
                        evidence_config,
                        self.base_config.seed + 20000,
                    )
                    split_aware_prior = branch.prior.split_aware()
                    split_aware_route = branch.split_aware_route()

                    for virtual_config in self.tuning_space.virtual_node_candidates:
                        augmented_scores, augmented_model = cross_fit_gnn(
                            inputs,
                            target_backbone,
                            folds,
                            self.base_config.gnn,
                            self.base_config.seed + 40000,
                            calibrated_prior=split_aware_prior,
                            route_mask=split_aware_route,
                            virtual_config=virtual_config,
                        )
                        matrices = split_aware_finalizer_matrices(
                            inputs.split,
                            graph_scores,
                            augmented_scores,
                            branch.prior,
                            branch.router,
                            branch.route_oof,
                            branch.route_full,
                        )

                        for finalizer_config in self.tuning_space.finalizer_candidates:
                            finalizer = fit_score_finalizer(
                                matrices.train,
                                inputs.labels[inputs.split.train],
                                matrices.dev,
                                inputs.labels[inputs.split.dev],
                                finalizer_config,
                                self.base_config.seed + 50000,
                            )
                            dev_probability = finalizer.model.predict_proba(matrices.dev)[:, 1]
                            metrics = binary_metrics(
                                inputs.labels[inputs.split.dev],
                                dev_probability,
                                finalizer.threshold,
                            )
                            key = _selection_key(metrics)
                            trial = TuningTrial(
                                router_name=router_candidate.name,
                                prior_name=prior_candidate.name,
                                route_threshold=float(route_threshold),
                                train_oof_route_coverage=float(
                                    np.mean(branch.route_oof[inputs.split.train])
                                ),
                                development_route_coverage=float(
                                    np.mean(branch.route_full[inputs.split.dev])
                                ),
                                virtual_node=virtual_config,
                                finalizer=finalizer_config,
                                selected_finalizer=finalizer.model_name,
                                decision_threshold=finalizer.threshold,
                                development_metrics=metrics,
                                selection_score=key[0],
                            )
                            trials.append(trial)
                            if best_key is None or key > best_key:
                                best_key = key
                                best_trial = trial
                                best_payload = _BestPayload(
                                    evidence_branch=branch,
                                    augmented_scores=augmented_scores,
                                    augmented_model=augmented_model,
                                    finalizer_matrices=matrices,
                                    finalizer=finalizer,
                                    development_probability=np.asarray(
                                        dev_probability, dtype=np.float32
                                    ),
                                )

        if best_payload is None or best_trial is None:
            raise RuntimeError("the Evi-VN tuning space produced no trial")
        test_probability = best_payload.finalizer.model.predict_proba(
            best_payload.finalizer_matrices.test
        )[:, 1]
        best_result = EviVNResult(
            hard_labels=hard_labels,
            evidence_branch=best_payload.evidence_branch,
            graph_scores=graph_scores,
            augmented_scores=best_payload.augmented_scores,
            finalizer_matrices=best_payload.finalizer_matrices,
            finalizer=best_payload.finalizer,
            development_probability=best_payload.development_probability,
            test_probability=np.asarray(test_probability, dtype=np.float32),
            graph_model=graph_model,
            augmented_model=best_payload.augmented_model,
            discovery_oof=discovery_oof,
        )
        return EviVNTuningResult(best_result, best_trial, tuple(trials))
