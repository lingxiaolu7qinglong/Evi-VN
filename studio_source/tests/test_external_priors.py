from __future__ import annotations

import json
import sys
from types import ModuleType

import numpy as np

from studio.llm_prior import LLMConnection, OpenAICompatiblePrior
from studio.registry import TabICLV2Estimator
from studio.data_io import make_demo_dataset
from studio.runner import TrainingSettings, train_evivn


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.body


def test_openai_compatible_prior_sends_key_and_parses_probabilities(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.headers.get("Authorization")
        observed["payload"] = json.loads(request.data)
        observed["timeout"] = timeout
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"probabilities":[0.2,0.8]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = OpenAICompatiblePrior(
        LLMConnection("fraud-model", "https://llm.example/v1", "secret-key"),
        seed=3,
    )
    model.fit(
        np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]]),
        np.asarray([0, 1, 0, 1]),
    )
    probability = model.predict_proba(np.asarray([[0.1, 0.9], [0.9, 0.1]]))

    assert observed["url"] == "https://llm.example/v1/chat/completions"
    assert observed["authorization"] == "Bearer secret-key"
    assert observed["payload"]["model"] == "fraud-model"
    assert observed["timeout"] == 120
    assert np.allclose(probability[:, 1], [0.2, 0.8])


def test_tabicl_adapter_preserves_the_probability_estimator_contract(monkeypatch) -> None:
    module = ModuleType("tabicl")

    class FakeTabICLClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, features, labels):
            self.rows = len(features)
            self.labels = np.asarray(labels)

        def predict_proba(self, features):
            positive = np.full(len(features), 0.65)
            return np.column_stack((1.0 - positive, positive))

    module.TabICLClassifier = FakeTabICLClassifier
    monkeypatch.setitem(sys.modules, "tabicl", module)
    estimator = TabICLV2Estimator(seed=9)
    estimator.fit(
        np.ones((6, 3)),
        np.asarray([0, 1, 0, 1, 0, 1]),
        sample_weight=np.ones(6),
    )

    assert np.allclose(estimator.predict_proba(np.ones((2, 3)))[:, 1], 0.65)
    assert estimator.model.kwargs["checkpoint_version"] == (
        "tabicl-classifier-v2-20260212.ckpt"
    )


def test_llm_prior_runs_through_the_split_aware_evivn_pipeline(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data)
        queries = json.loads(payload["messages"][1]["content"])["queries"]
        probability = [0.75 if row[0] >= 0 else 0.25 for row in queries]
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"probabilities": probability})
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    inputs, summary = make_demo_dataset(seed=11, node_count=48)
    run = train_evivn(
        inputs,
        summary,
        TrainingSettings(
            target_backbone="gcn",
            discovery_backbones=("mlp", "gcn"),
            router="logistic",
            prior="llm_openai_compatible",
            folds=2,
            hidden_dim=16,
            max_epochs=1,
            patience=1,
            oof_epochs=1,
            oof_patience=1,
            compact_tuning=False,
            llm_model="local-test-model",
            llm_endpoint="http://localhost:8080/v1",
            llm_api_key="temporary-test-key",
            llm_data_sharing_confirmed=True,
        ),
    )

    assert np.isfinite(run.result.test_probability).all()
    assert run.selected_configuration["llm_model"] == "local-test-model"
