# Evi-VN Core

## Install and Test

Python 3.11, from this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Train

Construct `EviVNInputs` from graph features, graph edges, evidence features,
binary labels, and a `Split(train, dev, test)`. Construct training folds as
`Fold(train, valid)`, with each training node held out once.

With prepared `inputs` and `folds`:

```python
from evivn import EviVN, EviVNTuner

result = EviVN().fit(inputs, folds, target_backbone="gcn")
fraud_probability = result.test_probability

tuned = EviVNTuner().tune(inputs, folds, target_backbone="gcn")
best_configuration = tuned.best_trial
fraud_probability = tuned.best_result.test_probability
```

The default discovery set contains 13 families. Pass `discovery_backbones` to
select the discovery set and `discovery_oof` to reuse stored OOF predictions.

`TuningSpace` defines router, prior, virtual node, and finalizer candidates.
Its default router grid is `0.01, 0.02, ..., 0.99`, selected by final
development Accuracy, with AUC and F1 as tie-breakers.

## Modules

| Module | Function |
| --- | --- |
| `config.py` | Backbone and model configuration |
| `types.py` | Input arrays, splits, folds, and score containers |
| `backbones.py` | GNN implementations |
| `evidence.py` | Consensus hard labels, router, prior, and calibration |
| `virtual_nodes.py` | Virtual class nodes, directed prior edges, and ProxyReg |
| `training.py` | Feature preprocessing and OOF GNN training |
| `finalizer.py` | Five-component finalization and evaluation metrics |
| `pipeline.py` | Evi-VN training pipeline |
| `tuning.py` | Configuration search |
