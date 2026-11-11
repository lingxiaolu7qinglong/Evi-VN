from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evivn import DISCOVERY_BACKBONES, logistic_estimator, xgboost_estimator
from evivn.evidence import ProbabilityEstimator


EstimatorFactory = Callable[[int], ProbabilityEstimator]


@dataclass(frozen=True)
class EstimatorEntry:
    key: str
    label: str
    factory: EstimatorFactory


def random_forest_estimator(seed: int) -> ProbabilityEstimator:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=4,
                ),
            ),
        ]
    )


def extra_trees_estimator(seed: int) -> ProbabilityEstimator:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=4,
                ),
            ),
        ]
    )


class TabICLV2Estimator:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.model: object | None = None

    def fit(
        self,
        features: object,
        labels: object,
        sample_weight: object | None = None,
        **_kwargs: object,
    ) -> TabICLV2Estimator:
        del sample_weight
        try:
            from tabicl import TabICLClassifier
        except ImportError as error:
            raise RuntimeError(
                "TabICLv2 is not installed. Use EviVN Studio WebSetup 0.2 or install tabicl."
            ) from error
        self.model = TabICLClassifier(
            checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
            random_state=self.seed,
            n_jobs=4,
            verbose=False,
        )
        self.model.fit(features, labels)
        return self

    def predict_proba(self, features: object) -> object:
        if self.model is None:
            raise RuntimeError("TabICLv2 must be fitted before prediction")
        return self.model.predict_proba(features)


def tabicl_v2_estimator(seed: int) -> ProbabilityEstimator:
    return TabICLV2Estimator(seed)


ESTIMATORS = (
    EstimatorEntry("xgboost", "XGBoost", xgboost_estimator),
    EstimatorEntry("logistic", "Logistic Regression", logistic_estimator),
    EstimatorEntry("random_forest", "Random Forest", random_forest_estimator),
    EstimatorEntry("extra_trees", "Extra Trees", extra_trees_estimator),
    EstimatorEntry("tabicl_v2", "TabICLv2", tabicl_v2_estimator),
)

LLM_PRIOR_KEY = "llm_openai_compatible"
LLM_PRIOR_LABEL = "LLM (OpenAI-compatible)"

BACKBONE_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "gat": "GAT",
    "sage": "GraphSAGE",
    "gcnii": "GCNII",
    "appnp": "APPNP",
    "gprgnn": "GPRGNN",
    "mixhop": "MixHop",
    "sign": "SIGN",
    "chebnet": "ChebNet",
    "graphconv": "GraphConv",
    "tagcn": "TAGCN",
    "gin": "GIN",
}


def estimator_entry(key: str) -> EstimatorEntry:
    for entry in ESTIMATORS:
        if entry.key == key:
            return entry
    raise KeyError(f"Unknown estimator: {key}")


def backbone_entries() -> tuple[tuple[str, str], ...]:
    return tuple((key, BACKBONE_LABELS[key]) for key in DISCOVERY_BACKBONES)
