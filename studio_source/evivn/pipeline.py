from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import DISCOVERY_BACKBONES, EviVNConfig
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
    fit_score_finalizer,
    split_aware_finalizer_matrices,
)
from .training import FittedNodeModel, cross_fit_gnn, discovery_oof_predictions
from .types import CrossFittedScores, EviVNInputs, Fold, validate_folds


@dataclass(frozen=True)
class EviVNResult:
    hard_labels: np.ndarray
    evidence_branch: EvidenceBranch
    graph_scores: CrossFittedScores
    augmented_scores: CrossFittedScores
    finalizer_matrices: FinalizerMatrices
    finalizer: FittedFinalizer
    development_probability: np.ndarray
    test_probability: np.ndarray
    graph_model: FittedNodeModel
    augmented_model: FittedNodeModel
    discovery_oof: dict[str, np.ndarray]


class EviVN:
    """Core Evi-VN training protocol on already prepared graph/evidence arrays."""

    def __init__(
        self,
        config: EviVNConfig | None = None,
        router_factory: EstimatorFactory = xgboost_estimator,
        prior_factory: EstimatorFactory = xgboost_estimator,
    ) -> None:
        self.config = EviVNConfig() if config is None else config
        self.router_factory = router_factory
        self.prior_factory = prior_factory

    def fit(
        self,
        inputs: EviVNInputs,
        folds: Sequence[Fold],
        target_backbone: str,
        discovery_backbones: Sequence[str] = DISCOVERY_BACKBONES,
        discovery_oof: dict[str, np.ndarray] | None = None,
    ) -> EviVNResult:
        inputs.validate()
        validate_folds(folds, inputs.split.train)
        target_backbone = target_backbone.lower()

        if discovery_oof is None:
            discovery_oof = discovery_oof_predictions(
                inputs,
                discovery_backbones,
                folds,
                self.config.gnn,
                self.config.seed,
            )
        hard_labels = consensus_hard_labels(
            discovery_oof,
            inputs.labels,
            inputs.split.train,
            self.config.hard_region,
        )
        branch = fit_evidence_branch(
            inputs,
            hard_labels,
            folds,
            self.router_factory,
            self.prior_factory,
            self.config.evidence,
            self.config.seed + 20000,
        )

        precomputed_target_oof = discovery_oof.get(target_backbone)
        graph_scores, graph_model = cross_fit_gnn(
            inputs,
            target_backbone,
            folds,
            self.config.gnn,
            self.config.seed + 30000,
            precomputed_oof=precomputed_target_oof,
        )

        split_aware_prior = branch.prior.split_aware()
        split_aware_route = branch.split_aware_route()
        augmented_scores, augmented_model = cross_fit_gnn(
            inputs,
            target_backbone,
            folds,
            self.config.gnn,
            self.config.seed + 40000,
            calibrated_prior=split_aware_prior,
            route_mask=split_aware_route,
            virtual_config=self.config.virtual_node,
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
        finalizer = fit_score_finalizer(
            matrices.train,
            inputs.labels[inputs.split.train],
            matrices.dev,
            inputs.labels[inputs.split.dev],
            self.config.finalizer,
            self.config.seed + 50000,
        )
        development_probability = finalizer.model.predict_proba(matrices.dev)[:, 1]
        test_probability = finalizer.model.predict_proba(matrices.test)[:, 1]
        return EviVNResult(
            hard_labels=hard_labels,
            evidence_branch=branch,
            graph_scores=graph_scores,
            augmented_scores=augmented_scores,
            finalizer_matrices=matrices,
            finalizer=finalizer,
            development_probability=np.asarray(development_probability, dtype=np.float32),
            test_probability=np.asarray(test_probability, dtype=np.float32),
            graph_model=graph_model,
            augmented_model=augmented_model,
            discovery_oof=discovery_oof,
        )
