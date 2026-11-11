from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    APPNP,
    ChebConv,
    GATConv,
    GCN2Conv,
    GCNConv,
    GINConv,
    GraphConv,
    MessagePassing,
    MixHopConv,
    SAGEConv,
    SGConv,
    TAGConv,
)
from torch_geometric.nn.conv.gcn_conv import gcn_norm


WEIGHTED_BACKBONES = {
    "gcn",
    "gcnii",
    "appnp",
    "gprgnn",
    "mixhop",
    "sign",
    "chebnet",
    "graphconv",
    "tagcn",
}


class GPRPropagation(MessagePassing):
    def __init__(self, steps: int = 10, alpha: float = 0.1) -> None:
        super().__init__(aggr="add")
        self.steps = steps
        coefficients = [alpha * (1.0 - alpha) ** step for step in range(steps)]
        coefficients.append((1.0 - alpha) ** steps)
        self.coefficients = nn.Parameter(torch.tensor(coefficients, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized_edges, weights = gcn_norm(
            edge_index,
            edge_weight=edge_weight,
            num_nodes=x.size(0),
            add_self_loops=True,
        )
        hidden = x
        output = self.coefficients[0] * hidden
        for step in range(1, self.steps + 1):
            hidden = self.propagate(normalized_edges, x=hidden, norm=weights)
            output = output + self.coefficients[step] * hidden
        return output

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class NodeClassifier(nn.Module):
    """The 13 first-stage families used by the discovery and target stages."""

    def __init__(
        self,
        family: str,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.family = family.lower()
        self.dropout = dropout

        if self.family == "mlp":
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 2)
        elif self.family == "gcn":
            self.layer1 = GCNConv(input_dim, hidden_dim)
            self.output = GCNConv(hidden_dim, 2)
        elif self.family == "gat":
            self.layer1 = GATConv(input_dim, hidden_dim // 4, heads=4, dropout=dropout)
            self.output = GATConv(hidden_dim, 2, heads=1, concat=False, dropout=dropout)
        elif self.family == "sage":
            self.layer1 = SAGEConv(input_dim, hidden_dim)
            self.output = SAGEConv(hidden_dim, 2)
        elif self.family == "gcnii":
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.gcnii1 = GCN2Conv(hidden_dim, alpha=0.1, theta=0.5, layer=1)
            self.gcnii2 = GCN2Conv(hidden_dim, alpha=0.1, theta=0.5, layer=2)
            self.output = nn.Linear(hidden_dim, 2)
        elif self.family == "appnp":
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 2)
            self.propagation = APPNP(K=10, alpha=0.1, dropout=dropout)
        elif self.family == "gprgnn":
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 2)
            self.propagation = GPRPropagation(steps=10, alpha=0.1)
        elif self.family == "mixhop":
            branch_dim = max(hidden_dim // 3, 1)
            mixed_dim = branch_dim * 3
            self.layer1 = MixHopConv(input_dim, branch_dim, powers=[0, 1, 2])
            self.mix_projection = nn.Linear(mixed_dim, hidden_dim)
            self.output = GCNConv(hidden_dim, 2)
        elif self.family == "sign":
            self.sign0 = nn.Linear(input_dim, hidden_dim)
            self.sign1 = SGConv(input_dim, hidden_dim, K=1, cached=False)
            self.sign2 = SGConv(input_dim, hidden_dim, K=2, cached=False)
            self.sign_projection = nn.Linear(3 * hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 2)
        elif self.family == "chebnet":
            self.layer1 = ChebConv(input_dim, hidden_dim, K=3)
            self.output = ChebConv(hidden_dim, 2, K=3)
        elif self.family == "graphconv":
            self.layer1 = GraphConv(input_dim, hidden_dim)
            self.output = GraphConv(hidden_dim, 2)
        elif self.family == "tagcn":
            self.layer1 = TAGConv(input_dim, hidden_dim, K=3)
            self.output = TAGConv(hidden_dim, 2, K=3)
        elif self.family == "gin":
            self.layer1 = GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.output = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2),
                )
            )
        else:
            raise ValueError(f"unsupported backbone family: {family}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.family == "mlp":
            hidden = self._drop(F.relu(self.layer1(x)))
            return self.output(hidden), hidden

        if self.family == "gcn":
            hidden = self._drop(F.relu(self.layer1(x, edge_index, edge_weight)))
            return self.output(hidden, edge_index, edge_weight), hidden

        if self.family == "gat":
            hidden = self._drop(F.elu(self.layer1(x, edge_index)))
            return self.output(hidden, edge_index), hidden

        if self.family == "sage":
            hidden = self._drop(F.relu(self.layer1(x, edge_index)))
            return self.output(hidden, edge_index), hidden

        if self.family == "gcnii":
            initial = F.relu(self.input_layer(x))
            hidden = self._drop(initial)
            hidden = self._drop(F.relu(self.gcnii1(hidden, initial, edge_index, edge_weight)))
            hidden = F.relu(self.gcnii2(hidden, initial, edge_index, edge_weight))
            return self.output(hidden), hidden

        if self.family == "appnp":
            hidden = self._drop(F.relu(self.layer1(x)))
            logits = self.output(hidden)
            return self.propagation(logits, edge_index, edge_weight), hidden

        if self.family == "gprgnn":
            hidden = self._drop(F.relu(self.layer1(x)))
            logits = self.output(hidden)
            return self.propagation(logits, edge_index, edge_weight), hidden

        if self.family == "mixhop":
            hidden = self.layer1(x, edge_index, edge_weight)
            hidden = self._drop(F.relu(self.mix_projection(hidden)))
            return self.output(hidden, edge_index, edge_weight), hidden

        if self.family == "sign":
            hidden = torch.cat(
                (
                    self.sign0(x),
                    self.sign1(x, edge_index, edge_weight),
                    self.sign2(x, edge_index, edge_weight),
                ),
                dim=1,
            )
            hidden = self._drop(F.relu(self.sign_projection(hidden)))
            return self.output(hidden), hidden

        if self.family in {"chebnet", "graphconv", "tagcn"}:
            hidden = self._drop(F.relu(self.layer1(x, edge_index, edge_weight)))
            return self.output(hidden, edge_index, edge_weight), hidden

        hidden = self._drop(F.relu(self.layer1(x, edge_index)))
        return self.output(hidden, edge_index), hidden

    def _drop(self, value: torch.Tensor) -> torch.Tensor:
        return F.dropout(value, p=self.dropout, training=self.training)
