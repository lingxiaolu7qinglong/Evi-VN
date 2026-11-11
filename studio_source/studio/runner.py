from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable

import numpy as np

from evivn import (
    EviVN,
    EviVNConfig,
    EviVNInputs,
    EviVNResult,
    EviVNTuner,
    FinalizerConfig,
    GNNConfig,
    HardRegionConfig,
    TuningSpace,
    VirtualNodeConfig,
    binary_metrics,
)
from evivn.tuning import EstimatorCandidate

from .data_io import DatasetSummary, make_folds
from .llm_prior import LLMConnection, llm_prior_factory
from .registry import LLM_PRIOR_KEY, EstimatorEntry, estimator_entry


LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrainingSettings:
    target_backbone: str
    discovery_backbones: tuple[str, ...]
    router: str = "xgboost"
    prior: str = "xgboost"
    folds: int = 3
    seed: int = 42
    hidden_dim: int = 64
    max_epochs: int = 80
    patience: int = 20
    oof_epochs: int = 55
    oof_patience: int = 12
    route_threshold: float = 0.07
    uncertainty_delta: float = 0.20
    tau_negative: float = 0.40
    tau_positive: float = 0.60
    edge_scale: float = 2.0
    compact_tuning: bool = True
    llm_model: str = ""
    llm_endpoint: str = ""
    llm_api_key: str = field(default="", repr=False, compare=False)
    llm_data_sharing_confirmed: bool = False


@dataclass(frozen=True)
class CompletedRun:
    run_id: str
    elapsed_seconds: float
    settings: TrainingSettings
    dataset: DatasetSummary
    result: EviVNResult
    development_metrics: dict[str, float]
    test_metrics: dict[str, float]
    hard_rate: float
    train_route_rate: float
    test_route_rate: float
    selected_configuration: dict[str, object]


def train_evivn(
    inputs: EviVNInputs,
    dataset: DatasetSummary,
    settings: TrainingSettings,
    log: LogCallback | None = None,
) -> CompletedRun:
    emit = log or (lambda _message: None)
    started = time.perf_counter()
    emit("Validating split-aware inputs")
    inputs.validate()
    folds = make_folds(inputs, settings.folds, settings.seed)

    router = estimator_entry(settings.router)
    prior = _prior_entry(settings)
    config = _make_config(settings)
    emit(
        f"Discovery: {len(settings.discovery_backbones)} backbones, "
        f"{settings.folds}-fold OOF"
    )
    emit(f"Evidence branch: {router.label} router, {prior.label} prior")

    selected: dict[str, object]
    if settings.compact_tuning:
        emit("Selecting the router threshold by final development accuracy")
        tuning_space = _make_tuning_space(settings, router, prior, config)
        tuned = EviVNTuner(config, tuning_space).tune(
            inputs,
            folds,
            target_backbone=settings.target_backbone,
            discovery_backbones=settings.discovery_backbones,
        )
        result = tuned.best_result
        selected = {
            "router": tuned.best_trial.router_name,
            "prior": tuned.best_trial.prior_name,
            "selection_objective": "development_accuracy",
            "route_threshold": tuned.best_trial.route_threshold,
            "train_oof_route_coverage": tuned.best_trial.train_oof_route_coverage,
            "development_route_coverage": (
                tuned.best_trial.development_route_coverage
            ),
            "tau_negative": tuned.best_trial.virtual_node.tau_negative,
            "tau_positive": tuned.best_trial.virtual_node.tau_positive,
            "edge_scale": tuned.best_trial.virtual_node.edge_scale,
            "finalizer": tuned.best_trial.selected_finalizer,
        }
    else:
        emit("Training original and virtual-node target GNNs")
        result = EviVN(
            config,
            router_factory=router.factory,
            prior_factory=prior.factory,
        ).fit(
            inputs,
            folds,
            target_backbone=settings.target_backbone,
            discovery_backbones=settings.discovery_backbones,
        )
        selected = {
            "router": router.label,
            "prior": prior.label,
            "selection_objective": "fixed_router_threshold",
            "route_threshold": result.evidence_branch.route_threshold,
            "train_oof_route_coverage": float(
                np.mean(result.evidence_branch.route_oof[inputs.split.train])
            ),
            "development_route_coverage": float(
                np.mean(result.evidence_branch.route_full[inputs.split.dev])
            ),
            "tau_negative": settings.tau_negative,
            "tau_positive": settings.tau_positive,
            "edge_scale": settings.edge_scale,
            "finalizer": result.finalizer.model_name,
        }
    if settings.prior == LLM_PRIOR_KEY:
        selected["llm_model"] = settings.llm_model
        selected["llm_endpoint"] = settings.llm_endpoint

    threshold = result.finalizer.threshold
    development_metrics = binary_metrics(
        inputs.labels[inputs.split.dev], result.development_probability, threshold
    )
    test_metrics = binary_metrics(
        inputs.labels[inputs.split.test], result.test_probability, threshold
    )
    train = inputs.split.train
    test = inputs.split.test
    elapsed = time.perf_counter() - started
    emit(f"Completed in {elapsed:.1f} seconds")
    return CompletedRun(
        run_id=datetime.now().strftime("%Y%m%d-%H%M%S"),
        elapsed_seconds=elapsed,
        settings=settings,
        dataset=dataset,
        result=result,
        development_metrics=development_metrics,
        test_metrics=test_metrics,
        hard_rate=float(np.mean(result.hard_labels[train])),
        train_route_rate=float(np.mean(result.evidence_branch.route_oof[train])),
        test_route_rate=float(np.mean(result.evidence_branch.route_full[test])),
        selected_configuration=selected,
    )


def _make_config(settings: TrainingSettings) -> EviVNConfig:
    return EviVNConfig(
        seed=settings.seed,
        hard_region=HardRegionConfig(uncertainty_delta=settings.uncertainty_delta),
        virtual_node=VirtualNodeConfig(
            tau_negative=settings.tau_negative,
            tau_positive=settings.tau_positive,
            edge_scale=settings.edge_scale,
        ),
        gnn=GNNConfig(
            hidden_dim=settings.hidden_dim,
            max_epochs=settings.max_epochs,
            patience=settings.patience,
            oof_max_epochs=settings.oof_epochs,
            oof_patience=settings.oof_patience,
        ),
        evidence=replace(
            EviVNConfig().evidence, route_threshold=settings.route_threshold
        ),
        finalizer=FinalizerConfig(tune_decision_threshold=False),
    )


def _make_tuning_space(
    settings: TrainingSettings,
    _router: object,
    _prior: object,
    config: EviVNConfig,
) -> TuningSpace:
    router_entry = estimator_entry(settings.router)
    prior_entry = _prior_entry(settings)
    base_proxy = config.virtual_node.proxy
    virtual_candidates = (
        config.virtual_node,
        VirtualNodeConfig(0.45, 0.55, 2.0, proxy=base_proxy),
        VirtualNodeConfig(0.35, 0.65, 4.0, proxy=base_proxy),
    )
    return TuningSpace(
        router_candidates=(EstimatorCandidate(router_entry.label, router_entry.factory),),
        prior_candidates=(EstimatorCandidate(prior_entry.label, prior_entry.factory),),
        virtual_node_candidates=virtual_candidates,
        finalizer_candidates=(config.finalizer,),
    )


def _prior_entry(settings: TrainingSettings) -> EstimatorEntry:
    if settings.prior != LLM_PRIOR_KEY:
        return estimator_entry(settings.prior)
    if not settings.llm_data_sharing_confirmed:
        raise ValueError("Confirm evidence sharing before using an LLM prior")
    connection = LLMConnection(
        model=settings.llm_model,
        endpoint=settings.llm_endpoint,
        api_key=settings.llm_api_key,
    )
    return EstimatorEntry(
        LLM_PRIOR_KEY,
        f"LLM ({connection.model})",
        llm_prior_factory(connection),
    )
