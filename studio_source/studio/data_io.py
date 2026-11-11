from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from evivn import EviVNInputs, Fold, Split


REQUIRED_ARRAYS = (
    "graph_features",
    "evidence_features",
    "edge_index",
    "labels",
    "train_idx",
    "dev_idx",
    "test_idx",
)


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    source: str
    nodes: int
    edges: int
    graph_dimensions: int
    evidence_dimensions: int
    positives: int
    train_nodes: int
    dev_nodes: int
    test_nodes: int

    @property
    def positive_rate(self) -> float:
        return self.positives / max(self.nodes, 1)


def summarize(inputs: EviVNInputs, name: str, source: str) -> DatasetSummary:
    return DatasetSummary(
        name=name,
        source=source,
        nodes=len(inputs.labels),
        edges=inputs.edge_index.shape[1],
        graph_dimensions=inputs.graph_features.shape[1],
        evidence_dimensions=inputs.evidence_features.shape[1],
        positives=int(np.sum(inputs.labels == 1)),
        train_nodes=len(inputs.split.train),
        dev_nodes=len(inputs.split.dev),
        test_nodes=len(inputs.split.test),
    )


def load_npz(path: str | Path) -> tuple[EviVNInputs, DatasetSummary]:
    file_path = Path(path).expanduser().resolve()
    with np.load(file_path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {', '.join(missing)}")
        edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
        if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.T
        edge_weight = (
            np.asarray(archive["edge_weight"], dtype=np.float32)
            if "edge_weight" in archive.files
            else None
        )
        inputs = EviVNInputs(
            graph_features=np.asarray(archive["graph_features"], dtype=np.float32),
            edge_index=edge_index,
            evidence_features=np.asarray(archive["evidence_features"], dtype=np.float32),
            labels=np.asarray(archive["labels"], dtype=np.int64).reshape(-1),
            split=Split(
                train=np.asarray(archive["train_idx"], dtype=np.int64).reshape(-1),
                dev=np.asarray(archive["dev_idx"], dtype=np.int64).reshape(-1),
                test=np.asarray(archive["test_idx"], dtype=np.int64).reshape(-1),
            ),
            edge_weight=edge_weight,
        )
    inputs.validate()
    _validate_training_requirements(inputs)
    return inputs, summarize(inputs, file_path.stem, str(file_path))


def make_folds(inputs: EviVNInputs, fold_count: int, seed: int) -> tuple[Fold, ...]:
    train = np.asarray(inputs.split.train, dtype=np.int64)
    labels = np.asarray(inputs.labels[train], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < fold_count:
        raise ValueError(
            f"Each training class needs at least {fold_count} nodes for {fold_count}-fold OOF"
        )
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    folds = []
    for train_position, valid_position in splitter.split(train, labels):
        folds.append(Fold(train=train[train_position], valid=train[valid_position]))
    return tuple(folds)


def make_demo_dataset(seed: int = 42, node_count: int = 96) -> tuple[EviVNInputs, DatasetSummary]:
    rng = np.random.default_rng(seed)
    labels = np.asarray([0, 1] * (node_count // 2), dtype=np.int64)
    rng.shuffle(labels)

    camouflage = rng.random(node_count) < 0.32
    observed_class = labels.copy()
    observed_class[camouflage] = 1 - observed_class[camouflage]

    graph_features = rng.normal(0.0, 0.75, size=(node_count, 8)).astype(np.float32)
    graph_features[:, 0] += 1.15 * (2 * observed_class - 1)
    graph_features[:, 1] += 0.45 * (2 * labels - 1)
    graph_features[:, 2] += camouflage.astype(np.float32) * 0.35

    evidence_features = rng.normal(0.0, 0.65, size=(node_count, 10)).astype(np.float32)
    evidence_features[:, 0] += 1.4 * (2 * labels - 1)
    evidence_features[:, 1] += 1.1 * camouflage.astype(np.float32)
    evidence_features[:, 2] += (observed_class != labels).astype(np.float32)
    evidence_features[:, 3] += 0.6 * (2 * labels - 1) * camouflage.astype(np.float32)

    edges: set[tuple[int, int]] = set()
    for node_id in range(node_count):
        preferred = np.flatnonzero(observed_class == observed_class[node_id])
        preferred = preferred[preferred != node_id]
        other = np.flatnonzero(observed_class != observed_class[node_id])
        for neighbor in rng.choice(preferred, size=min(4, len(preferred)), replace=False):
            edges.add((node_id, int(neighbor)))
            edges.add((int(neighbor), node_id))
        for neighbor in rng.choice(other, size=min(1, len(other)), replace=False):
            edges.add((node_id, int(neighbor)))
            edges.add((int(neighbor), node_id))
    edge_index = np.asarray(sorted(edges), dtype=np.int64).T

    train, dev, test = _stratified_split(labels, rng)
    inputs = EviVNInputs(
        graph_features=graph_features,
        edge_index=edge_index,
        evidence_features=evidence_features,
        labels=labels,
        split=Split(train=train, dev=dev, test=test),
    )
    inputs.validate()
    return inputs, summarize(inputs, "Camouflage demo", "Built-in synthetic graph")


def save_npz_template(path: str | Path, inputs: EviVNInputs) -> None:
    values: dict[str, np.ndarray] = {
        "graph_features": inputs.graph_features,
        "evidence_features": inputs.evidence_features,
        "edge_index": inputs.edge_index,
        "labels": inputs.labels,
        "train_idx": inputs.split.train,
        "dev_idx": inputs.split.dev,
        "test_idx": inputs.split.test,
    }
    if inputs.edge_weight is not None:
        values["edge_weight"] = inputs.edge_weight
    np.savez_compressed(Path(path), **values)


def _stratified_split(
    labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    partitions = [[], [], []]
    for class_id in (0, 1):
        members = np.flatnonzero(labels == class_id)
        rng.shuffle(members)
        first = int(round(0.60 * len(members)))
        second = int(round(0.80 * len(members)))
        partitions[0].extend(members[:first].tolist())
        partitions[1].extend(members[first:second].tolist())
        partitions[2].extend(members[second:].tolist())
    return tuple(np.asarray(sorted(part), dtype=np.int64) for part in partitions)  # type: ignore[return-value]


def _validate_training_requirements(inputs: EviVNInputs) -> None:
    for name, indices in (
        ("training", inputs.split.train),
        ("development", inputs.split.dev),
        ("test", inputs.split.test),
    ):
        if len(indices) == 0:
            raise ValueError(f"The {name} split cannot be empty")
        if len(np.unique(inputs.labels[indices])) < 2:
            raise ValueError(f"The {name} split must contain both classes")
