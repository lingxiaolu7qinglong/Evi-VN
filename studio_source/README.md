# EviVN Studio

Desktop application for graph fraud detection with Evi-VN.

## Install and Run

Python 3.11, from this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Workflow

1. Import prepared arrays using [NPZ_SCHEMA.md](NPZ_SCHEMA.md), or select **Load demo**.
2. Select discovery backbones, target backbone, router, and prior generator.
3. Configure training and run Evi-VN.
4. View Accuracy, AUC, F1, and routing diagnostics.
5. Export a ZIP containing `manifest.json` and `test_scores.npz`.

Available router and prior models are XGBoost, Logistic Regression, Random
Forest, Extra Trees, and TabICL v2. For an LLM prior, enter the model name,
API base URL, and API key in the model settings and enable evidence sharing.

Configuration search selects the router threshold from `0.01` through `0.99`
using final development Accuracy. Fixed configuration training uses the
specified threshold.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe app.py --self-test self-test.json
```

The [Windows installer](https://github.com/lingxiaolu7qinglong/Evi-VN/releases/download/v0.2.2/EviVNStudio-WebSetup-0.2.2.exe)
installs the runtime, CPU dependencies, and TabICL v2 checkpoint on first launch.
