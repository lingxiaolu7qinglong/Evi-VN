from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .backbones import NodeClassifier, WEIGHTED_BACKBONES
from .config import GNNConfig, VirtualNodeConfig
from .types import CrossFittedScores, EviVNInputs, Fold, validate_folds
from .virtual_nodes import (
    build_augmented_graph,
    class_weights,
    edges_for_backbone,
    proxy_regularization,
)


@dataclass(frozen=True)
class FittedNodeModel:
    model: NodeClassifier
    probability: np.ndarray
    best_epoch: int
    imputer: SimpleImputer
    scaler: StandardScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _preprocess_graph_features(
    features: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(features[fit_indices])
    imputed = imputer.transform(features)
    scaler.fit(imputed[fit_indices])
    return scaler.transform(imputed).astype(np.float32), imputer, scaler


def _safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probability))


def _inner_split(
    indices: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 10 or min(np.bincount(labels[indices], minlength=2)) < 2:
        return indices, None
    inner_train, inner_dev = train_test_split(
        indices,
        test_size=fraction,
        random_state=seed,
        stratify=labels[indices],
    )
    return np.asarray(inner_train), np.asarray(inner_dev)


def fit_node_model(
    inputs: EviVNInputs,
    family: str,
    train_indices: np.ndarray,
    config: GNNConfig,
    seed: int,
    validation_indices: np.ndarray | None = None,
    calibrated_prior: np.ndarray | None = None,
    route_mask: np.ndarray | None = None,
    virtual_config: VirtualNodeConfig | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> FittedNodeModel:
    set_seed(seed)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = (
        None if validation_indices is None else np.asarray(validation_indices, dtype=np.int64)
    )
    features, imputer, scaler = _preprocess_graph_features(inputs.graph_features, train_indices)
    use_virtual = calibrated_prior is not None or route_mask is not None or virtual_config is not None
    if use_virtual and (calibrated_prior is None or route_mask is None or virtual_config is None):
        raise ValueError("prior, route mask, and virtual config must be supplied together")

    if use_virtual:
        augmented = build_augmented_graph(
            features,
            inputs.edge_index,
            inputs.labels,
            train_indices,
            calibrated_prior,
            route_mask,
            virtual_config,
            inputs.edge_weight,
        )
        model_features = augmented.features
        model_edges, model_weights = edges_for_backbone(augmented, family)
        virtual_ids = augmented.virtual_ids
    else:
        model_features = features
        model_edges = np.asarray(inputs.edge_index, dtype=np.int64)
        model_weights = (
            np.ones(model_edges.shape[1], dtype=np.float32)
            if inputs.edge_weight is None
            else np.asarray(inputs.edge_weight, dtype=np.float32)
        )
        if family.lower() not in WEIGHTED_BACKBONES:
            model_weights = None
        virtual_ids = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.as_tensor(model_features, dtype=torch.float32, device=device)
    edge_tensor = torch.as_tensor(model_edges, dtype=torch.long, device=device)
    edge_weight_tensor = (
        None
        if model_weights is None
        else torch.as_tensor(model_weights, dtype=torch.float32, device=device)
    )
    labels_tensor = torch.as_tensor(inputs.labels, dtype=torch.long, device=device)
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    loss_weight = class_weights(inputs.labels, train_indices, device)

    model = NodeClassifier(
        family,
        model_features.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_limit = config.max_epochs if max_epochs is None else int(max_epochs)
    early_stop = config.patience if patience is None else int(patience)
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -np.inf
    best_epoch = epoch_limit
    stale_epochs = 0

    for epoch in range(1, epoch_limit + 1):
        model.train()
        optimizer.zero_grad()
        logits, hidden = model(x_tensor, edge_tensor, edge_weight_tensor)
        classification = F.cross_entropy(
            logits[train_tensor],
            labels_tensor[train_tensor],
            weight=loss_weight,
        )
        loss = classification
        if virtual_ids is not None:
            regularization, _ = proxy_regularization(
                hidden,
                labels_tensor,
                train_tensor,
                virtual_ids,
                loss_weight,
                virtual_config.proxy,
            )
            warmup = virtual_config.proxy.warmup_epochs
            eta = 1.0 if warmup == 0 else min(1.0, epoch / warmup)
            loss = loss + eta * regularization
        loss.backward()
        optimizer.step()

        if validation_indices is None:
            continue
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(x_tensor, edge_tensor, edge_weight_tensor)[0], dim=1)[
                : len(inputs.labels), 1
            ].cpu().numpy()
        current_auc = _safe_auc(
            inputs.labels[validation_indices], probability[validation_indices]
        )
        if current_auc > best_auc + 1e-5:
            best_auc = current_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(x_tensor, edge_tensor, edge_weight_tensor)[0], dim=1)[
            : len(inputs.labels), 1
        ].cpu().numpy().astype(np.float32)
    return FittedNodeModel(model, probability, best_epoch, imputer, scaler)


def _fit_oof_fold(
    inputs: EviVNInputs,
    family: str,
    fold: Fold,
    config: GNNConfig,
    seed: int,
    calibrated_prior: np.ndarray | None = None,
    route_mask: np.ndarray | None = None,
    virtual_config: VirtualNodeConfig | None = None,
) -> FittedNodeModel:
    inner_train, inner_dev = _inner_split(
        fold.train,
        inputs.labels,
        config.inner_dev_fraction,
        seed,
    )
    selected = fit_node_model(
        inputs,
        family,
        inner_train,
        config,
        seed + 100,
        validation_indices=inner_dev,
        calibrated_prior=calibrated_prior,
        route_mask=route_mask,
        virtual_config=virtual_config,
        max_epochs=config.oof_max_epochs,
        patience=config.oof_patience,
    )
    if inner_dev is None:
        return selected
    return fit_node_model(
        inputs,
        family,
        fold.train,
        config,
        seed + 200,
        calibrated_prior=calibrated_prior,
        route_mask=route_mask,
        virtual_config=virtual_config,
        max_epochs=selected.best_epoch,
        patience=config.oof_patience,
    )


def cross_fit_gnn(
    inputs: EviVNInputs,
    family: str,
    folds: Sequence[Fold],
    config: GNNConfig,
    seed: int,
    calibrated_prior: np.ndarray | None = None,
    route_mask: np.ndarray | None = None,
    virtual_config: VirtualNodeConfig | None = None,
    precomputed_oof: np.ndarray | None = None,
) -> tuple[CrossFittedScores, FittedNodeModel]:
    validate_folds(folds, inputs.split.train)
    if precomputed_oof is None:
        oof = np.full(len(inputs.labels), np.nan, dtype=np.float32)
        for fold_id, fold in enumerate(folds):
            fitted = _fit_oof_fold(
                inputs,
                family,
                fold,
                config,
                seed + 1000 + fold_id * 10,
                calibrated_prior=calibrated_prior,
                route_mask=route_mask,
                virtual_config=virtual_config,
            )
            oof[fold.valid] = fitted.probability[fold.valid]
    else:
        oof = np.asarray(precomputed_oof, dtype=np.float32).copy()

    full_model = fit_node_model(
        inputs,
        family,
        inputs.split.train,
        config,
        seed + 9000,
        validation_indices=inputs.split.dev,
        calibrated_prior=calibrated_prior,
        route_mask=route_mask,
        virtual_config=virtual_config,
    )
    scores = CrossFittedScores(oof, full_model.probability, inputs.split.train)
    return scores, full_model


def discovery_oof_predictions(
    inputs: EviVNInputs,
    families: Sequence[str],
    folds: Sequence[Fold],
    config: GNNConfig,
    seed: int,
) -> dict[str, np.ndarray]:
    predictions = {}
    for family_id, family in enumerate(families):
        oof = np.full(len(inputs.labels), np.nan, dtype=np.float32)
        for fold_id, fold in enumerate(folds):
            fitted = _fit_oof_fold(
                inputs,
                family,
                fold,
                config,
                seed + family_id * 10000 + fold_id * 10,
            )
            oof[fold.valid] = fitted.probability[fold.valid]
        if not np.isfinite(oof[inputs.split.train]).all():
            raise RuntimeError(f"incomplete discovery OOF predictions for {family}")
        predictions[family.lower()] = oof
    return predictions
