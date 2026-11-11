from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evivn import EviVNInputs

from .runner import CompletedRun


def export_run(path: str | Path, inputs: EviVNInputs, run: CompletedRun) -> Path:
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")

    public_settings = asdict(run.settings)
    api_key_configured = bool(public_settings.pop("llm_api_key", ""))
    public_settings["llm_api_key_configured"] = api_key_configured
    manifest = {
        "format": "EviVN Studio run package",
        "version": 2,
        "run_id": run.run_id,
        "dataset": asdict(run.dataset),
        "settings": public_settings,
        "selected_configuration": run.selected_configuration,
        "development_metrics": run.development_metrics,
        "test_metrics": run.test_metrics,
        "diagnostics": {
            "hard_rate": run.hard_rate,
            "train_route_rate": run.train_route_rate,
            "test_route_rate": run.test_route_rate,
            "elapsed_seconds": run.elapsed_seconds,
        },
        "score_order": [
            "augmented_gnn_logit",
            "calibrated_prior_logit",
            "original_gnn_logit",
            "router_score",
            "route_mask",
        ],
    }
    score_buffer = io.BytesIO()
    test = inputs.split.test
    np.savez_compressed(
        score_buffer,
        node_id=test,
        label=inputs.labels[test],
        final_probability=run.result.test_probability,
        original_probability=run.result.graph_scores.full_train[test],
        augmented_probability=run.result.augmented_scores.full_train[test],
        calibrated_prior=run.result.evidence_branch.prior.full_train[test],
        router_score=run.result.evidence_branch.router.full_train[test],
        route_mask=run.result.evidence_branch.route_full[test].astype(np.int8),
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("test_scores.npz", score_buffer.getvalue())
    return output
