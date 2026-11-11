from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    dev: np.ndarray
    test: np.ndarray

    def validate(self, node_count: int) -> None:
        arrays = [np.asarray(part, dtype=np.int64) for part in (self.train, self.dev, self.test)]
        joined = np.concatenate(arrays)
        if len(np.unique(joined)) != len(joined):
            raise ValueError("train, dev, and test indices must be disjoint")
        if len(joined) and (joined.min() < 0 or joined.max() >= node_count):
            raise ValueError("split index is outside the graph")


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class EviVNInputs:
    graph_features: np.ndarray
    edge_index: np.ndarray
    evidence_features: np.ndarray
    labels: np.ndarray
    split: Split
    edge_weight: np.ndarray | None = None

    def validate(self) -> None:
        node_count = len(self.labels)
        if self.graph_features.ndim != 2 or len(self.graph_features) != node_count:
            raise ValueError("graph_features must have shape [num_nodes, graph_dim]")
        if self.evidence_features.ndim != 2 or len(self.evidence_features) != node_count:
            raise ValueError("evidence_features must have shape [num_nodes, evidence_dim]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= node_count):
            raise ValueError("edge_index contains an invalid node id")
        if self.edge_weight is not None and len(self.edge_weight) != self.edge_index.shape[1]:
            raise ValueError("edge_weight must align with edge_index")
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("Evi-VN currently expects binary labels 0/1")
        self.split.validate(node_count)


@dataclass(frozen=True)
class CrossFittedScores:
    train_oof: np.ndarray
    full_train: np.ndarray
    train_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.train_oof.shape != self.full_train.shape:
            raise ValueError("OOF and full-training scores must have the same shape")
        train = np.asarray(self.train_indices, dtype=np.int64)
        if not np.isfinite(self.train_oof[train]).all():
            raise ValueError("OOF scores are incomplete on training nodes")
        if not np.isfinite(self.full_train).all():
            raise ValueError("full-training scores are incomplete")

    def split_aware(self) -> np.ndarray:
        scores = np.asarray(self.full_train, dtype=np.float32).copy()
        scores[self.train_indices] = self.train_oof[self.train_indices]
        return scores


def validate_folds(folds: Sequence[Fold], train_indices: np.ndarray) -> None:
    expected = np.sort(np.asarray(train_indices, dtype=np.int64))
    held_out = []
    expected_set = set(expected.tolist())
    for fold in folds:
        fold_train = np.asarray(fold.train, dtype=np.int64)
        fold_valid = np.asarray(fold.valid, dtype=np.int64)
        if len(np.unique(fold_train)) != len(fold_train):
            raise ValueError("fold training indices cannot contain duplicates")
        if len(np.unique(fold_valid)) != len(fold_valid):
            raise ValueError("fold validation indices cannot contain duplicates")
        if np.intersect1d(fold_train, fold_valid).size:
            raise ValueError("a fold cannot train on its held-out nodes")
        if not set(fold_train.tolist()).issubset(expected_set):
            raise ValueError("fold training indices must belong to the training split")
        if not set(fold_valid.tolist()).issubset(expected_set):
            raise ValueError("fold validation indices must belong to the training split")
        expected_fold_train = expected_set.difference(fold_valid.tolist())
        if set(fold_train.tolist()) != expected_fold_train:
            raise ValueError("each fold must train on the full T excluding its held-out fold")
        held_out.append(fold_valid)
    observed = np.sort(np.concatenate(held_out)) if held_out else np.empty(0, dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("OOF folds must hold out every training node exactly once")
