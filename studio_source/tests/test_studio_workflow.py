from __future__ import annotations

import json
import zipfile

import numpy as np

from studio.data_io import load_npz, make_demo_dataset, save_npz_template
from studio.exporter import export_run
from evivn import EviVNTuner
from studio.registry import estimator_entry
from studio.runner import (
    TrainingSettings,
    _make_config,
    _make_tuning_space,
    train_evivn,
)


def test_npz_round_trip_preserves_feature_isolation(tmp_path) -> None:
    inputs, _ = make_demo_dataset(seed=3, node_count=48)
    path = tmp_path / "dataset.npz"
    save_npz_template(path, inputs)
    loaded, summary = load_npz(path)

    assert np.array_equal(loaded.graph_features, inputs.graph_features)
    assert np.array_equal(loaded.evidence_features, inputs.evidence_features)
    assert not np.shares_memory(loaded.graph_features, loaded.evidence_features)
    assert summary.graph_dimensions == inputs.graph_features.shape[1]
    assert summary.evidence_dimensions == inputs.evidence_features.shape[1]


def test_training_and_export_workflow(tmp_path) -> None:
    inputs, summary = make_demo_dataset(seed=5, node_count=48)
    settings = TrainingSettings(
        target_backbone="gcn",
        discovery_backbones=("mlp", "gcn"),
        router="logistic",
        prior="logistic",
        folds=2,
        hidden_dim=16,
        max_epochs=1,
        patience=1,
        oof_epochs=1,
        oof_patience=1,
        compact_tuning=False,
        llm_api_key="must-not-be-exported",
    )
    run = train_evivn(inputs, summary, settings)
    package = export_run(tmp_path / "run.zip", inputs, run)

    assert np.isfinite(run.result.test_probability).all()
    assert set(run.test_metrics) == {"accuracy", "auc", "f1", "threshold"}
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {"manifest.json", "test_scores.npz"}
        manifest = json.loads(archive.read("manifest.json"))
        assert "must-not-be-exported" not in archive.read("manifest.json").decode()
        assert "llm_api_key" not in manifest["settings"]
        assert manifest["settings"]["llm_api_key_configured"] is True
        assert manifest["score_order"] == [
            "augmented_gnn_logit",
            "calibrated_prior_logit",
            "original_gnn_logit",
            "router_score",
            "route_mask",
        ]


def test_studio_router_search_uses_final_development_accuracy() -> None:
    settings = TrainingSettings(
        target_backbone="gcn",
        discovery_backbones=("mlp", "gcn"),
        router="logistic",
        prior="logistic",
    )
    config = _make_config(settings)
    router = estimator_entry(settings.router)
    prior = estimator_entry(settings.prior)
    space = _make_tuning_space(settings, router, prior, config)

    assert settings.compact_tuning is True
    assert EviVNTuner(config, space).selection_metric == "accuracy"
    assert space.route_thresholds == tuple(index / 100 for index in range(1, 100))
