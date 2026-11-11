# Evi-VN

**Hard Region Guided Virtual Node Evidence Injection for GNN-Based Fraud Detection**

Jiran Tao, Yifan Wu, Binyan Jiang

The Hong Kong Polytechnic University

Evi-VN learns hard regions shared across GNNs and injects complementary evidence
through virtual class nodes to improve fraud detection. The framework combines
OOF discovery, a hard node router, a calibrated prior generator, ProxyReg, and a
five-component prediction finalizer.

## Download

- [EviVN Studio 0.2.2 for Windows](https://github.com/lingxiaolu7qinglong/Evi-VN/releases/download/v0.2.2/EviVNStudio-WebSetup-0.2.2.exe)
- [Installer SHA-256](https://github.com/lingxiaolu7qinglong/Evi-VN/releases/download/v0.2.2/EviVNStudio-WebSetup-0.2.2.sha256)
- [Source code](https://github.com/lingxiaolu7qinglong/Evi-VN/archive/refs/tags/v0.2.2.zip)

## Repository

| Directory | Contents |
| --- | --- |
| [code/EviVN_code](code/EviVN_code) | Evi-VN model, configuration search, and core tests |
| [studio_source](studio_source) | Desktop application, NPZ import, result export, and tests |
| [software](software) | Windows download links and installer checksum |

## Windows Application

Run the Windows installer. The first launch installs the Python runtime, CPU
dependencies, and the TabICL v2 classifier checkpoint over the internet.

1. Import an NPZ dataset or select **Load demo**.
2. Choose discovery backbones, a target backbone, a router, and a prior generator.
3. Run training and view Accuracy, AUC, F1, and routing results.
4. Export the run configuration, metrics, and node predictions.

For an LLM prior, select **LLM (OpenAI-compatible)** and enter the model name,
API base URL, and API key. Enable evidence sharing to use the configured service.

## Run from Source

Python 3.11:

```powershell
cd studio_source
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Run the application tests and the demo workflow:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe app.py --self-test self-test.json
```

## Input Format

Studio imports prepared NPZ arrays containing `graph_features`,
`evidence_features`, `edge_index`, `labels`, `train_idx`, `dev_idx`, and
`test_idx`. Labels are `0` for benign nodes and `1` for fraud nodes.

GNNs use `graph_features` and graph edges. The router and prior generator use
the separate `evidence_features` matrix. Numerical evidence can represent
structured records, text, images, or audio.

See [NPZ_SCHEMA.md](studio_source/NPZ_SCHEMA.md) for array shapes.

## Models and Configuration

Discovery and target families: MLP, GCN, GAT, GraphSAGE, GCNII, APPNP, GPRGNN,
MixHop, SIGN, ChebNet, GraphConv, TAGCN, and GIN.

Studio router and prior models: XGBoost, Logistic Regression, Random Forest,
Extra Trees, and TabICL v2. LLM connections are available for prior generation.

`EviVNTuner` searches router thresholds from `0.01` to `0.99` in steps of
`0.01`. Configurations are selected by final development Accuracy, followed
by AUC and F1 as tie-breakers. `EviVN.fit` runs a specified configuration.

See [the core API](code/EviVN_code/README.md) for programmatic training.
