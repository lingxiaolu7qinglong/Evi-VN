# EviVN Studio NPZ schema

Archives are loaded with `numpy.load(..., allow_pickle=False)` and must contain:

| Array | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `graph_features` | `[N, Dg]` | float | Data-side features available to GNN backbones |
| `evidence_features` | `[N, De]` | float | Feature-isolated evidence used by router and prior only |
| `edge_index` | `[2, E]` or `[E, 2]` | int | Directed source-target graph edges |
| `labels` | `[N]` | int | Binary labels, negative `0`, positive/fraud `1` |
| `train_idx` | `[Nt]` | int | Real training nodes |
| `dev_idx` | `[Nd]` | int | Development-only model selection nodes |
| `test_idx` | `[Ns]` | int | Locked reporting nodes |

Optional `edge_weight` has shape `[E]`. Split indices must be disjoint and every
split must contain both classes. Evidence arrays must exclude labels, split IDs,
GNN predictions, consensus-hard labels, and downstream decisions.
