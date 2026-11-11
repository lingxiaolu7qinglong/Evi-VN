from __future__ import annotations

import numpy as np
import pytest
import torch

import evivn.evidence as evidence
import evivn.training as training
from evivn import (
    DISCOVERY_BACKBONES,
    EvidenceConfig,
    EstimatorCandidate,
    EviVN,
    EviVNConfig,
    EviVNInputs,
    EviVNTuner,
    FinalizerConfig,
    Fold,
    GNNConfig,
    Split,
    TuningSpace,
    VirtualNodeConfig,
    logistic_estimator,
)
from evivn.backbones import NodeClassifier
from evivn.config import ProxyRegConfig
from evivn.evidence import consensus_hard_labels
from evivn.finalizer import finalizer_features, split_aware_finalizer_matrices
from evivn.tuning import _selection_key
from evivn.types import CrossFittedScores, validate_folds
from evivn.virtual_nodes import (
    build_augmented_graph,
    edges_for_backbone,
    proxy_regularization,
)


def test_default_router_threshold_grid_covers_full_probability_range() -> None:
    thresholds = TuningSpace().route_thresholds
    assert thresholds == tuple(step / 100 for step in range(1, 100))


def test_source_only_virtual_edges_and_class_prototypes() -> None:
    features = np.asarray([[0.0], [2.0], [10.0], [12.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    edges = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    prior = np.asarray([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
    route = np.asarray([True, True, True, False])
    graph = build_augmented_graph(
        features,
        edges,
        labels,
        np.arange(4),
        prior,
        route,
        VirtualNodeConfig(edge_scale=2.0),
    )

    negative_id, positive_id = graph.virtual_ids
    virtual_edges = graph.edge_index[:, graph.base_edge_count :]
    assert virtual_edges.tolist() == [[negative_id, positive_id], [0, 2]]
    assert np.all(virtual_edges[0] >= len(labels))
    assert np.all(virtual_edges[1] < len(labels))
    assert np.allclose(graph.edge_weight[-2:], [1.6, 1.6])
    assert np.allclose(graph.features[negative_id], [1.0, 1.0, 0.0])
    assert np.allclose(graph.features[positive_id], [11.0, 0.0, 1.0])


def test_consensus_hard_region_uses_wrong_or_uncertain_votes() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    discovery = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.9],
            [0.49, 0.51, 0.48],
            [0.9, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    hard = consensus_hard_labels(
        discovery,
        labels,
        np.arange(4),
        EviVNConfig().hard_region,
    )
    assert hard.tolist() == [0, 1, 1, 0]


def test_anchor_center_is_stop_gradient() -> None:
    hidden = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [0.1, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    train = torch.arange(4)
    weight = torch.ones(2)
    config = ProxyRegConfig(
        lambda_proxy=0.0,
        lambda_separation=0.0,
        lambda_anchor=1.0,
        anchor_radius=0.0,
    )
    loss, _ = proxy_regularization(hidden, labels, train, (4, 5), weight, config)
    loss.backward()
    assert torch.allclose(hidden.grad[:4], torch.zeros_like(hidden.grad[:4]))
    assert torch.linalg.vector_norm(hidden.grad[4:]) > 0


def test_all_discovery_backbones_accept_the_augmented_graph() -> None:
    features = np.asarray([[i, i % 2] for i in range(8)], dtype=np.float32)
    labels = np.asarray([0, 1] * 4, dtype=np.int64)
    source = np.arange(8)
    target = np.roll(source, -1)
    edges = np.vstack((np.concatenate((source, target)), np.concatenate((target, source))))
    graph = build_augmented_graph(
        features,
        edges,
        labels,
        np.arange(8),
        np.asarray([0.1, 0.9] * 4, dtype=np.float32),
        np.ones(8, dtype=bool),
        VirtualNodeConfig(edge_scale=2.0),
    )
    for family in DISCOVERY_BACKBONES:
        model_edges, model_weights = edges_for_backbone(graph, family)
        model = NodeClassifier(family, graph.features.shape[1], hidden_dim=12, dropout=0.0)
        logits, hidden = model(
            torch.tensor(graph.features),
            torch.tensor(model_edges),
            None if model_weights is None else torch.tensor(model_weights),
        )
        assert logits.shape == (10, 2)
        assert hidden.shape[0] == 10


def test_finalizer_uses_oof_training_scores_and_full_dev_test_scores() -> None:
    train = np.asarray([0, 1])
    split = Split(train, np.asarray([2]), np.asarray([3]))

    def scores(offset: float) -> CrossFittedScores:
        oof = np.asarray([0.1 + offset, 0.2 + offset, np.nan, np.nan], dtype=np.float32)
        full = np.asarray([0.8, 0.8, 0.3 + offset, 0.4 + offset], dtype=np.float32)
        return CrossFittedScores(oof, full, train)

    matrices = split_aware_finalizer_matrices(
        split,
        scores(0.00),
        scores(0.05),
        scores(0.10),
        scores(0.15),
        np.asarray([False, True, False, False]),
        np.asarray([True, True, True, False]),
    )
    assert matrices.train.shape == (2, 5)
    assert matrices.dev.shape == (1, 5)
    assert matrices.test.shape == (1, 5)
    assert matrices.train[0, 4] == 0.0
    assert matrices.dev[0, 4] == 1.0
    assert not np.isclose(matrices.train[0, 2], np.log(0.8 / 0.2))


def test_complete_evivn_pipeline_runs_from_prepared_arrays() -> None:
    node_count = 14
    graph_features = np.asarray(
        [[i / node_count, (i % 3) / 2] for i in range(node_count)], dtype=np.float32
    )
    evidence_features = np.asarray(
        [[i % 2, (i % 4) / 3] for i in range(node_count)], dtype=np.float32
    )
    labels = np.asarray([0, 1] * 7, dtype=np.int64)
    source = np.arange(node_count)
    target = np.roll(source, -1)
    edge_index = np.vstack(
        (np.concatenate((source, target)), np.concatenate((target, source)))
    ).astype(np.int64)
    split = Split(np.arange(10), np.asarray([10, 11]), np.asarray([12, 13]))
    folds = (
        Fold(np.arange(5, 10), np.arange(5)),
        Fold(np.arange(5), np.arange(5, 10)),
    )
    inputs = EviVNInputs(
        graph_features,
        edge_index,
        evidence_features,
        labels,
        split,
    )
    config = EviVNConfig(
        gnn=GNNConfig(
            hidden_dim=8,
            dropout=0.0,
            learning_rate=0.01,
            weight_decay=0.0,
            max_epochs=2,
            patience=1,
            oof_max_epochs=2,
            oof_patience=1,
            inner_dev_fraction=0.2,
        ),
        evidence=EvidenceConfig(route_threshold=0.07),
        finalizer=FinalizerConfig(logistic_c=(1.0,), include_xgboost=False),
    )
    result = EviVN(
        config,
        router_factory=logistic_estimator,
        prior_factory=logistic_estimator,
    ).fit(
        inputs,
        folds,
        target_backbone="gcn",
        discovery_backbones=("mlp", "gcn"),
    )
    assert result.finalizer_matrices.train.shape == (10, 5)
    assert result.development_probability.shape == (2,)
    assert result.test_probability.shape == (2,)
    assert np.isfinite(result.test_probability).all()


def test_main_model_tuner_selects_only_from_development_metrics() -> None:
    node_count = 14
    graph_features = np.asarray(
        [[i / node_count, (i % 3) / 2] for i in range(node_count)], dtype=np.float32
    )
    evidence_features = np.asarray(
        [[i % 2, (i % 4) / 3] for i in range(node_count)], dtype=np.float32
    )
    labels = np.asarray([0, 1] * 7, dtype=np.int64)
    source = np.arange(node_count)
    target = np.roll(source, -1)
    edge_index = np.vstack(
        (np.concatenate((source, target)), np.concatenate((target, source)))
    ).astype(np.int64)
    split = Split(np.arange(10), np.asarray([10, 11]), np.asarray([12, 13]))
    folds = (
        Fold(np.arange(5, 10), np.arange(5)),
        Fold(np.arange(5), np.arange(5, 10)),
    )
    inputs = EviVNInputs(
        graph_features,
        edge_index,
        evidence_features,
        labels,
        split,
    )
    config = EviVNConfig(
        gnn=GNNConfig(
            hidden_dim=8,
            dropout=0.0,
            learning_rate=0.01,
            weight_decay=0.0,
            max_epochs=1,
            patience=1,
            oof_max_epochs=1,
            oof_patience=1,
            inner_dev_fraction=0.2,
        ),
        evidence=EvidenceConfig(route_threshold=0.07),
        finalizer=FinalizerConfig(logistic_c=(1.0,), include_xgboost=False),
    )
    virtual_candidates = (
        VirtualNodeConfig(tau_negative=0.4, tau_positive=0.6, edge_scale=1.0),
    )
    space = TuningSpace(
        router_candidates=(EstimatorCandidate("logistic", logistic_estimator),),
        prior_candidates=(EstimatorCandidate("logistic", logistic_estimator),),
        route_thresholds=(0.05, 0.10),
        virtual_node_candidates=virtual_candidates,
        finalizer_candidates=(config.finalizer,),
    )
    discovery_oof = np.full(node_count, np.nan, dtype=np.float32)
    discovery_oof[split.train] = np.linspace(0.1, 0.9, len(split.train))
    result = EviVNTuner(config, space).tune(
        inputs,
        folds,
        target_backbone="gcn",
        discovery_oof={"gcn": discovery_oof},
    )

    assert len(result.trials) == 2
    assert EviVNTuner(config, space).selection_metric == "accuracy"
    assert result.best_trial in result.trials
    assert result.best_trial.selection_score == max(
        trial.selection_score for trial in result.trials
    )
    assert result.best_trial.selection_score == result.best_trial.development_metrics[
        "accuracy"
    ]
    assert result.best_trial.route_threshold in space.route_thresholds
    assert result.best_trial.development_route_coverage == pytest.approx(
        np.mean(result.best_result.evidence_branch.route_full[split.dev])
    )
    assert result.best_result.test_probability.shape == (2,)


def test_router_threshold_selection_prioritizes_final_development_accuracy() -> None:
    higher_accuracy = {"accuracy": 0.90, "auc": 0.60, "f1": 0.60}
    higher_auc_and_f1 = {"accuracy": 0.89, "auc": 0.99, "f1": 0.99}

    assert _selection_key(higher_accuracy) > _selection_key(higher_auc_and_f1)


def test_folds_must_train_on_exactly_t_minus_held_out_fold() -> None:
    train = np.arange(6)
    invalid = (
        Fold(np.asarray([2, 3, 4]), np.asarray([0, 1])),
        Fold(np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
        Fold(np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
    )
    with pytest.raises(ValueError, match="full T"):
        validate_folds(invalid, train)


def test_oof_epoch_selection_refits_on_the_complete_fold_train(monkeypatch) -> None:
    node_count = 24
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    inputs = EviVNInputs(
        graph_features=np.ones((node_count, 2), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        evidence_features=np.ones((node_count, 2), dtype=np.float32),
        labels=labels,
        split=Split(np.arange(20), np.asarray([20, 21]), np.asarray([22, 23])),
    )
    fold = Fold(train=np.arange(5, 20), valid=np.arange(5))
    calls = []

    def fake_fit(
        inputs,
        family,
        train_indices,
        config,
        seed,
        validation_indices=None,
        **kwargs,
    ):
        calls.append(
            {
                "train": np.asarray(train_indices),
                "validation": validation_indices,
                "max_epochs": kwargs.get("max_epochs"),
            }
        )
        return type(
            "Fitted",
            (),
            {
                "probability": np.full(node_count, 0.5, dtype=np.float32),
                "best_epoch": 3,
            },
        )()

    monkeypatch.setattr(training, "fit_node_model", fake_fit)
    training._fit_oof_fold(inputs, "gcn", fold, GNNConfig(), seed=42)

    assert len(calls) == 2
    assert len(calls[0]["train"]) < len(fold.train)
    assert calls[0]["validation"] is not None
    assert np.array_equal(calls[1]["train"], fold.train)
    assert calls[1]["validation"] is None
    assert calls[1]["max_epochs"] == 3


def test_only_the_prior_logit_is_bounded_in_finalizer_features() -> None:
    probability = np.asarray([1.0 - 1e-9], dtype=np.float64)
    features = finalizer_features(
        probability,
        probability,
        probability,
        np.asarray([0.9]),
        np.asarray([True]),
    )
    assert features[0, 0] > 8.0
    assert features[0, 1] == 8.0
    assert features[0, 2] > 8.0


def test_prior_models_fit_only_routed_training_nodes(monkeypatch) -> None:
    node_count = 16
    train = np.arange(12)
    inputs = EviVNInputs(
        graph_features=np.ones((node_count, 2), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
        evidence_features=np.column_stack(
            (np.arange(node_count), np.linspace(0.05, 0.95, node_count))
        ).astype(np.float32),
        labels=np.asarray([(index // 2) % 2 for index in range(node_count)]),
        split=Split(train, np.asarray([12, 13]), np.asarray([14, 15])),
    )
    folds = tuple(
        Fold(
            train=np.setdiff1d(train, np.arange(offset, len(train), 3)),
            valid=np.arange(offset, len(train), 3),
        )
        for offset in range(3)
    )
    hard = np.zeros(node_count, dtype=np.int64)
    hard[train] = (train >= 6).astype(np.int64)
    prior_fit_indices = []

    class DeterministicEstimator:
        def predict_proba(self, features):
            positive = np.clip(features[:, 1], 1e-3, 1.0 - 1e-3)
            return np.column_stack((1.0 - positive, positive))

    def router_factory(seed):
        return DeterministicEstimator()

    def prior_factory(seed):
        return DeterministicEstimator()

    def fake_fit(factory, seed, features, labels, indices):
        if factory is prior_factory:
            prior_fit_indices.append(np.asarray(indices).copy())
        return factory(seed)

    monkeypatch.setattr(evidence, "_fit_estimator", fake_fit)
    branch = evidence.fit_evidence_branch(
        inputs,
        hard,
        folds,
        router_factory,
        prior_factory,
        EvidenceConfig(route_threshold=0.5),
        seed=42,
    )

    expected = [fold.train[branch.route_oof[fold.train]] for fold in folds]
    expected.append(train[branch.route_oof[train]])
    assert len(prior_fit_indices) == len(expected)
    for observed, routed in zip(prior_fit_indices, expected):
        assert np.array_equal(observed, routed)


def test_calibration_excludes_each_held_out_fold(monkeypatch) -> None:
    train = np.arange(6)
    folds = (
        Fold(np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
        Fold(np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
        Fold(np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
    )
    raw_oof = np.asarray([0.11, 0.22, 0.33, 0.44, 0.55, 0.66], dtype=np.float32)
    raw_full = np.asarray([0.71, 0.72, 0.73, 0.74, 0.75, 0.76], dtype=np.float32)
    targets = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    fitted_probabilities = []

    class IdentityCalibrator:
        def fit(self, probability, labels):
            fitted_probabilities.append(np.asarray(probability).copy())
            return self

        def transform(self, probability):
            return np.asarray(probability, dtype=np.float32)

    monkeypatch.setattr(evidence, "PlattCalibrator", IdentityCalibrator)
    scores = evidence._cross_fitted_calibration(
        raw_oof,
        raw_full,
        targets,
        train,
        folds,
    )

    for fold, fitted in zip(folds, fitted_probabilities[: len(folds)]):
        assert np.array_equal(fitted, raw_oof[fold.train])
        assert not np.isin(raw_oof[fold.valid], fitted).any()
    assert np.array_equal(scores.train_oof[train], raw_oof)
    assert np.array_equal(scores.full_train, raw_full)
