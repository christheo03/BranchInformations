# Machine Learning Branch Predictor using Static Branch Information

A pipeline for predicting conditional branch outcomes using machine learning models trained on **static** program features (control-flow structure, opcodes, register usage), rather than the runtime branch history that conventional hardware predictors rely on. Branch execution traces are collected with an Intel Pin tool and enriched with control-flow-graph features from `angr`, then used to train MLP classifiers that predict whether a branch is Highly Taken, Highly Not-Taken, or Not Biased (or, for the `Taken_NotTaken` models, a per-branch taken probability).

## Pipeline overview

1. **Trace collection** (`data_collector/branch_behavor.cpp`) — a Pin tool instruments a binary, records every conditional branch's execution/taken counts, its neighboring instructions, and the flag-writing instruction that feeds it, and dumps everything to `branches.csv`.
2. **Static enrichment** (`data_collector/angrversion.py`) — loads the binary into `angr`, rebuilds its CFG, and extends `branches.csv` with dominance, loop, and call/store information for each branch's taken and fall-through targets (writing the result back to `branches.csv`).
3. **Feature engineering & training** (`mlp_predictor/`) — the enriched CSVs (one per benchmark, expected under `../../results/<benchmark>.csv` relative to `mlp_predictor/`) are loaded, labeled, encoded, and used to train embedding-based MLP models in PyTorch.
4. **Evaluation** — trained models are scored on held-out benchmarks, reporting accuracy, macro F1, and predicted miss rate.

## Repository structure

```
BranchInformations/
├── data_collector/
│   ├── branch_behavor.cpp    # Intel Pin tool: collects per-branch execution traces -> branches.csv
│   └── angrversion.py        # Enriches branches.csv with angr-derived CFG/static features
└── mlp_predictor/
    ├── ml_configs.py         # Central config: paths, benchmark lists, column groups, hyperparameters
    ├── data_engin.py         # Data loading, labeling, feature encoding/normalization
    ├── mlp_model.py          # Model definitions (embedding MLP, tanh-output MLPs) + train/eval helpers
    ├── hb_branch_pred.py     # Trains/evaluates the 3-class (HT / HNT / NB) branch predictor
    ├── LOO_method.py         # Leave-one-benchmark-out cross-validation over hb_branch_pred's model
    └── Taken_NotTaken/
        ├── criterion.py      # Custom loss functions (weighted/static miss-rate, accuracy loss)
        ├── EMB_CT.py          # Trains the embedding-based taken/not-taken model (Optuna-tuned)
        ├── ESP.py             # Trains a StandardScaler-normalized MLP on the ESP feature set
        ├── SS_CT.py           # Trains a StandardScaler-normalized MLP on the CT feature set
        └── Evaluation/
            ├── helper.py      # Loads saved checkpoints, rebuilds/normalizes inputs for scoring
            └── model-eval.py  # Scores saved SS_CT / ESP / EMB_CT models and reports miss rates
```

Some files referenced by the workflow (`UBD_NU.cpp`, `merge_csv.py`, `makefile`, `get_dcfg_heuristics.py`, the `results/` and `Validation/` data directories, saved model checkpoints under `Taken_NotTaken/Models/`) are intentionally untracked — see `.gitignore`.

## Branch classes

Branches are labeled from their observed taken rate (`Taken / Executed`), using a threshold of 0.5% by default:

| Label | Meaning | Condition |
|---|---|---|
| HNT | Highly Not-Taken | taken rate < 0.5% |
| NB | Not Biased | 0.5% ≤ taken rate ≤ 99.5% |
| HT | Highly Taken | taken rate > 99.5% |

The `Taken_NotTaken` models instead predict a continuous taken probability per branch and are scored with weighted/static miss-rate losses (`Taken_NotTaken/criterion.py`).

## Requirements

**Trace collection / enrichment** (`data_collector/`):
- [Intel Pin](https://software.intel.com/sitewide-tools/pin) (for `branch_behavor.cpp`, requires `pin.H`)
- Python 3 with `angr`, `networkx`, `pyvex`

**ML pipeline** (`mlp_predictor/`):
- Python 3 with `torch`, `pandas`, `numpy`, `scikit-learn`
- `optuna` (for the `Taken_NotTaken` training scripts, which run hyperparameter search)

Install the Python dependencies with:

```bash
pip install angr networkx pyvex torch pandas numpy scikit-learn optuna
```

## Usage

**1. Collect a branch trace** (requires a Pin-compiled `.so`/`.dll` from `branch_behavor.cpp`):

```bash
pin -t branch_behavor.so -- ./your_benchmark [args]
# produces branches.csv in the working directory
```

**2. Enrich the trace with static CFG features:**

```bash
python angrversion.py path/to/binary
# updates branches.csv in place
```

Repeat for each benchmark and place the resulting CSVs under `results/<benchmark_name>.csv` (as referenced by `DATA_DIR` in `mlp_predictor/ml_configs.py`).

**3. Train and evaluate the 3-class predictor:**

```bash
cd mlp_predictor
python hb_branch_pred.py
```

**4. Run leave-one-benchmark-out cross-validation:**

```bash
python LOO_method.py
```

**5. Train a Taken/Not-Taken model** (run as a module so the relative `criterion` import resolves):

```bash
python -m Taken_NotTaken.EMB_CT
python -m Taken_NotTaken.ESP
python -m Taken_NotTaken.SS_CT
```

**6. Evaluate saved Taken/Not-Taken checkpoints** (expects `.pth` files under `Taken_NotTaken/Models/`):

```bash
cd Taken_NotTaken/Evaluation
python model-eval.py
```

## Configuration

Benchmark lists, train/test splits, feature column groups, and training hyperparameters (embedding size, epochs, learning rate, early-stopping patience, batch size) are all centralized in `mlp_predictor/ml_configs.py` — adjust `DATA_DIR`, `ALL_FILES`, and `TEST_FILES` there to point at your own benchmark set.
