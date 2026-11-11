from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from evivn.evidence import ProbabilityEstimator


@dataclass(frozen=True)
class LLMConnection:
    model: str
    endpoint: str
    api_key: str
    timeout_seconds: int = 120

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("LLM model is required")
        parts = urlsplit(self.endpoint.strip())
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("LLM endpoint must be an HTTP or HTTPS URL")
        if parts.username or parts.password:
            raise ValueError("Do not place credentials in the LLM endpoint URL")


class OpenAICompatiblePrior:
    def __init__(
        self,
        connection: LLMConnection,
        seed: int,
        max_context_rows: int = 16,
        max_features: int = 96,
        batch_size: int = 8,
    ) -> None:
        connection.validate()
        self.connection = connection
        self.seed = seed
        self.max_context_rows = max_context_rows
        self.max_features = max_features
        self.batch_size = batch_size
        self.feature_indices: np.ndarray | None = None
        self.medians: np.ndarray | None = None
        self.examples: list[dict[str, object]] = []

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
        **_kwargs: object,
    ) -> OpenAICompatiblePrior:
        del sample_weight
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if matrix.ndim != 2 or len(matrix) != len(targets):
            raise ValueError("LLM prior expects a two-dimensional feature matrix")
        if len(np.unique(targets)) < 2:
            raise ValueError("LLM prior requires examples from both classes")

        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        medians = np.nanmedian(finite, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        variance = np.var(matrix, axis=0)
        count = min(matrix.shape[1], self.max_features)
        self.feature_indices = np.argsort(variance)[-count:]
        self.feature_indices.sort()
        self.medians = medians
        matrix = matrix[:, self.feature_indices]

        rng = np.random.default_rng(self.seed)
        per_class = max(1, self.max_context_rows // 2)
        selected: list[int] = []
        for class_id in (0, 1):
            members = np.flatnonzero(targets == class_id)
            if len(members) > per_class:
                members = rng.choice(members, size=per_class, replace=False)
            selected.extend(int(index) for index in members)
        selected.sort()
        self.examples = [
            {
                "features": self._serialize_row(matrix[index]),
                "label": int(targets[index]),
            }
            for index in selected
        ]
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = self._transform(features)
        positive: list[float] = []
        for start in range(0, len(matrix), self.batch_size):
            batch = matrix[start : start + self.batch_size]
            positive.extend(self._predict_batch(batch))
        probability = np.clip(np.asarray(positive, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        return np.column_stack((1.0 - probability, probability))

    def _transform(self, features: np.ndarray) -> np.ndarray:
        if self.feature_indices is None or self.medians is None or not self.examples:
            raise RuntimeError("LLM prior must be fitted before prediction")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.medians):
            raise ValueError("LLM prior feature dimensions changed after fitting")
        matrix = np.where(np.isfinite(matrix), matrix, self.medians)
        return matrix[:, self.feature_indices]

    def _predict_batch(self, matrix: np.ndarray) -> list[float]:
        queries = [self._serialize_row(row) for row in matrix]
        user_payload = {
            "labeled_examples": self.examples,
            "queries": queries,
        }
        request_payload = {
            "model": self.connection.model.strip(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Estimate binary fraud probabilities from structured evidence-chain "
                        "feature vectors. Label 1 means fraud and label 0 means benign. "
                        "Return JSON only as {\"probabilities\":[p1,...]} in query order."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload, separators=(",", ":")),
                },
            ],
        }
        raw = json.dumps(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.connection.api_key:
            headers["Authorization"] = f"Bearer {self.connection.api_key}"
        request = urllib.request.Request(
            _chat_completions_url(self.connection.endpoint),
            data=raw,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.connection.timeout_seconds
            ) as response:
                body = response.read(2_000_000).decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM endpoint returned HTTP {error.code}: {detail[:500]}"
            ) from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM endpoint request failed: {error.reason}") from None

        try:
            response_payload = json.loads(body)
            content = response_payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            probabilities = _parse_probabilities(str(content), len(matrix))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("LLM endpoint returned an unsupported response") from error
        return probabilities

    @staticmethod
    def _serialize_row(row: np.ndarray) -> list[float]:
        return [float(f"{value:.6g}") for value in row]


def llm_prior_factory(connection: LLMConnection) -> Callable[[int], ProbabilityEstimator]:
    connection.validate()

    def factory(seed: int) -> ProbabilityEstimator:
        return OpenAICompatiblePrior(connection, seed)

    return factory


def _chat_completions_url(endpoint: str) -> str:
    parts = urlsplit(endpoint.strip())
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = f"{path}/chat/completions"
    else:
        final_path = f"{path}/v1/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, final_path, parts.query, ""))


def _parse_probabilities(content: str, expected: int) -> list[float]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("LLM response did not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    values = payload.get("probabilities")
    if not isinstance(values, list) or len(values) != expected:
        raise RuntimeError(
            f"LLM response must contain {expected} probabilities in query order"
        )
    probabilities = np.asarray(values, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise RuntimeError("LLM probabilities must be finite values between zero and one")
    return probabilities.tolist()
