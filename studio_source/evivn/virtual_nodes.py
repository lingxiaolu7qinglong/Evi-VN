from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .backbones import WEIGHTED_BACKBONES
from .config import ProxyRegConfig, VirtualNodeConfig


@dataclass(frozen=True)
class AugmentedGraph:
    features: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    virtual_ids: tuple[int, int]
    base_edge_count: int


def build_augmented_graph(
    graph_features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    prototype_indices: np.ndarray,
    calibrated_prior: np.ndarray,
    route_mask: np.ndarray,
    config: VirtualNodeConfig,
    base_edge_weight: np.ndarray | None = None,
) -> AugmentedGraph:
    node_count, feature_dim = graph_features.shape
    prototype_indices = np.asarray(prototype_indices, dtype=np.int64)
    calibrated_prior = np.asarray(calibrated_prior, dtype=np.float32)
    route_mask = np.asarray(route_mask, dtype=bool)

    if len(calibrated_prior) != node_count or len(route_mask) != node_count:
        raise ValueError("prior and route mask must align with real nodes")
    if not np.isfinite(calibrated_prior).all() or not ((calibrated_prior >= 0) & (calibrated_prior <= 1)).all():
        raise ValueError("calibrated prior must contain finite probabilities")

    real_features = np.pad(graph_features, ((0, 0), (0, 2)))
    prototypes = []
    for class_id in (0, 1):
        members = prototype_indices[labels[prototype_indices] == class_id]
        if not len(members):
            raise ValueError(f"prototype training split has no class {class_id}")
        prototype = np.pad(graph_features[members].mean(axis=0), (0, 2))
        prototype[feature_dim + class_id] = 1.0
        prototypes.append(prototype)

    negative_id, positive_id = node_count, node_count + 1
    features = np.vstack((real_features, prototypes)).astype(np.float32)
    virtual_edges = []
    virtual_weights = []
    for node_id in range(node_count):
        if not route_mask[node_id]:
            continue
        prior = float(calibrated_prior[node_id])
        if prior >= config.tau_positive:
            source = positive_id
        elif prior <= config.tau_negative:
            source = negative_id
        else:
            continue
        virtual_edges.append((source, node_id))
        virtual_weights.append(config.edge_scale * 2.0 * abs(prior - 0.5))

    base_edge_count = edge_index.shape[1]
    if virtual_edges:
        full_edges = np.hstack((edge_index, np.asarray(virtual_edges, dtype=np.int64).T))
    else:
        full_edges = np.asarray(edge_index, dtype=np.int64).copy()
    if base_edge_weight is None:
        base_weights = np.ones(base_edge_count, dtype=np.float32)
    else:
        base_weights = np.asarray(base_edge_weight, dtype=np.float32)
    full_weights = np.concatenate((base_weights, np.asarray(virtual_weights, dtype=np.float32)))
    return AugmentedGraph(features, full_edges, full_weights, (negative_id, positive_id), base_edge_count)


def edges_for_backbone(
    graph: AugmentedGraph,
    family: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if family.lower() in WEIGHTED_BACKBONES:
        return graph.edge_index, graph.edge_weight
    if family.lower() == "mlp":
        return graph.edge_index, None

    base_edges = graph.edge_index[:, : graph.base_edge_count]
    virtual_edges = graph.edge_index[:, graph.base_edge_count :]
    if not virtual_edges.shape[1]:
        return base_edges, None
    repeats = np.rint(graph.edge_weight[graph.base_edge_count :]).astype(np.int64)
    repeats = np.maximum(repeats, 1)
    repeated_virtual = np.repeat(virtual_edges, repeats, axis=1)
    return np.hstack((base_edges, repeated_virtual)), None


def class_weights(labels: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels[indices], minlength=2).astype(np.float32)
    weights = len(indices) / np.maximum(2.0 * counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def proxy_regularization(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    virtual_ids: tuple[int, int],
    class_weight: torch.Tensor,
    config: ProxyRegConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized = F.normalize(hidden, p=2, dim=1)
    proxies = normalized[list(virtual_ids)]
    train_hidden = normalized[train_indices]
    train_labels = labels[train_indices]

    similarities = train_hidden @ proxies.T / config.temperature
    proxy_loss = F.cross_entropy(similarities, train_labels, weight=class_weight)
    separation_loss = F.relu(
        torch.dot(proxies[0], proxies[1]) - config.cosine_margin
    ).square()

    anchor_loss = hidden.new_zeros(())
    for class_id in (0, 1):
        members = train_indices[train_labels == class_id]
        if members.numel() == 0:
            continue
        class_center = normalized[members].mean(dim=0).detach()
        distance = torch.linalg.vector_norm(proxies[class_id] - class_center)
        anchor_loss = anchor_loss + F.relu(distance - config.anchor_radius).square()

    weighted = (
        config.lambda_proxy * proxy_loss
        + config.lambda_separation * separation_loss
        + config.lambda_anchor * anchor_loss
    )
    return weighted, {
        "proxy": proxy_loss,
        "separation": separation_loss,
        "anchor": anchor_loss,
    }
