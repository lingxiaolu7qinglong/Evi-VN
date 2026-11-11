from __future__ import annotations

from dataclasses import dataclass, field


DISCOVERY_BACKBONES = (
    "mlp",
    "gcn",
    "gat",
    "sage",
    "gcnii",
    "appnp",
    "gprgnn",
    "mixhop",
    "sign",
    "chebnet",
    "graphconv",
    "tagcn",
    "gin",
)


@dataclass(frozen=True)
class HardRegionConfig:
    uncertainty_delta: float = 0.20
    error_fraction: float = 0.50
    hard_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty_delta <= 0.5:
            raise ValueError("uncertainty_delta must be in [0, 0.5]")
        if not 0.0 < self.error_fraction <= 1.0:
            raise ValueError("error_fraction must be in (0, 1]")
        if not 0.0 < self.hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ProxyRegConfig:
    lambda_proxy: float = 0.10
    lambda_separation: float = 0.01
    lambda_anchor: float = 0.003
    temperature: float = 0.20
    cosine_margin: float = 0.0
    anchor_radius: float = 0.75
    warmup_epochs: int = 30

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.anchor_radius < 0.0:
            raise ValueError("anchor_radius must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")


@dataclass(frozen=True)
class VirtualNodeConfig:
    tau_negative: float = 0.40
    tau_positive: float = 0.60
    edge_scale: float = 1.0
    proxy: ProxyRegConfig = field(default_factory=ProxyRegConfig)

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_negative < 0.5:
            raise ValueError("tau_negative must be below 0.5")
        if not 0.5 < self.tau_positive <= 1.0:
            raise ValueError("tau_positive must be above 0.5")
        if self.edge_scale <= 0.0:
            raise ValueError("edge_scale must be positive")


@dataclass(frozen=True)
class GNNConfig:
    hidden_dim: int = 128
    dropout: float = 0.35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 350
    patience: int = 45
    oof_max_epochs: int = 220
    oof_patience: int = 30
    inner_dev_fraction: float = 0.20


@dataclass(frozen=True)
class EvidenceConfig:
    route_threshold: float = 0.07

    def __post_init__(self) -> None:
        if not 0.0 <= self.route_threshold <= 1.0:
            raise ValueError("route_threshold must be in [0, 1]")


@dataclass(frozen=True)
class FinalizerConfig:
    logistic_c: tuple[float, ...] = (0.03, 0.10, 0.30, 1.0, 3.0)
    include_xgboost: bool = True
    tune_decision_threshold: bool = False


@dataclass(frozen=True)
class EviVNConfig:
    seed: int = 42
    hard_region: HardRegionConfig = field(default_factory=HardRegionConfig)
    virtual_node: VirtualNodeConfig = field(default_factory=VirtualNodeConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    finalizer: FinalizerConfig = field(default_factory=FinalizerConfig)
