# Citation

This work was submitted to the *AI* (MDPI) Special Issue Integrating Large Language Models into Robotic Autonomy (https://www.mdpi.com/journal/ai/special_issues/26397176N2) and is under review.

# PSTAR A Unified One-Shot Autoregressive Model for Path Planning

**Abstract:** Autonomous systems require path planners that execute within strict real-time constraints. Classical grid search algorithms guarantee optimality but waste compute by recalculating routes from scratch without leveraging past spatial experience. To bypass runtime search entirely, we propose a search-free planning framework that generates collision-free grid routes in a single autoregressive rollout. Built on a PerceiverAR architecture trained on A* optimal routes, our model tokenizes 2D grid contexts, obstacles, and target coordinates into a unified spatial representation to predict end to end paths. To avoid wasted compute on impossible queries, we pair path generation with an upfront, XGBoost-based route-availability gate that identifies unsolvable maps before model rollout. Evaluated across diverse obstacle densities, the route availability gate achieved  99.34\% accuracy, while the transformer generator produces optimal routes in 99.66\% of solvable test cases. Together, this yields a unified framework pipeline that immediately filters infeasible requests while delivering constant-time, near-optimal path generation.

**Keywords:** Path Planning; Transformer; PerceiverAR; Grid Navigation; Constant-Time Inference; Autoregressive path generation; Imitation Learning

---

## Overview

This repository is the **model and training** half of the pipeline. It consumes the datasets
produced by the companion repository
[`ASTAR_Dataset_Generation`](../ASTAR_Dataset_Generation) and contains three components:

* **PSTAR model** (`PSTARModel/`) — the PerceiverAR Transformer backbone, the direction-model
  wrapper that emits U/D/R/L logits and computes the loss, the autoregressive training loop,
  the inference rollout, and route evaluation/visualization.
* **RAGate gate** (`RAGateModel/`) — feature extraction from the trained Transformer plus three
  interchangeable route-availability classifiers (LDA, MLP, XGBoost) and a comparison report.
* **Shared** (`shared/`) — the streaming dataloader, the unified inference framework
  (gate + rollout), and common utilities.

A single, fully commented configuration file drives all behavior: `config.py`.

## Requirements

* Python 3.10+
* An NVIDIA GPU with CUDA is strongly recommended (training on CPU is supported but slow).

Install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA 12.4 build of PyTorch (`torch==2.6.0+cu124` and matching
`torchvision`/`torchaudio`) alongside `numpy`, `pandas`, `matplotlib`, `ijson`, `einops`,
`scikit-learn`, and `xgboost`. Adjust the Torch index URL / version in `requirements.txt` if you
need a different CUDA runtime. `main.py` also checks for missing packages at startup and installs
them automatically.

## Data

Before training, generate the datasets with the companion repository and place them under a
`Datasets/` folder at the project root, matching the paths in `config.py`:

```
Datasets/
  PSTAR/                     # transformer training data (solvable grids)
    train/  val/  test/      # train_sol / val_sol / test_sol  (.pkl, .json, .pt)
  RAGate/                    # gate classifier data (solvable + unsolvable grids)
    train/  val/  test/      # RAGate_train / RAGate_val / RAGate_test  (.pkl, .json, .pt)
```

The transformer trains on the `sol` (solvable) splits; the gate trains on the RAGate splits,
which contain both solvable (`label=1`) and unsolvable (`label=0`) grids.

## Instructions

The pipeline stage(s) that run are selected by boolean switches near the top of `config.py`.
Set the stage(s) you want to `True`, review the matching settings, then run:

```bash
python main.py
```

### Pipeline switches (in `config.py`)

```python
"pstar_training":    True,    # train the PSTAR transformer
"inference":         False,   # run the autoregressive rollout on the test set (no gate)
"ragate_training":   False,   # extract embeddings + train/evaluate the RAGate classifiers
"unified_framework": False,   # run the gate + rollout together on the RAGate test split
```

### Model and training settings (in `config.py`)

```python
"grid_size":              (10, 10),
"context_matrix_length":  100,     # one token per grid cell
"foundroute_matrix_length": 75,    # max route (latent) length
"sequence_len":           175,     # context(100) + latent(75)

"num_unique_tokens":      4,       # direction vocabulary: U, D, R, L
"dim":                    512,     # embedding dimension
"num_layers":             12,
"heads":                  8,
"causal":                 True,    # autoregressive attention

"batch":                  128,
"epochs":                 200,
"learning_rate":          2e-4,
"optimizer":              "adam",
"training_mode":          "autoregressive",
"file_format":            "tensor_pt",   # "tensor_pt" | "json" | "pickle"
```

Feature toggles at the top of `config.py` control the per-cell feature vector. Features f0–f3
(cell index, x, y, status) are always on; `feature_dx_dy`, `feature_free_neighbours`, and
`feature_is_interior` add optional features. **`feature_size` is computed automatically — do not
set it by hand.** At startup, the pipeline prints a diagnostics banner showing the active device
(GPU/CPU) and the exact feature layout for the run.

Checkpoints and results are written under `Results/PSTAR/` and `Results/RAGate/`. To resume
training from the last checkpoint, set `resume_from_checkpoint` in `config.py`.

### Training the RAGate gate

With `ragate_training: True`, `main.py` runs the full gate workflow in order: extract context
embeddings from the trained Transformer, then train and evaluate the LDA, MLP, and XGBoost
classifiers, and finally emit a side-by-side comparison. Select the deployed classifier and its
decision threshold with `ragate_model` (`"lda"`/`"mlp"`/`"xgboost"`) and `ragate_model_tau1`.

The gate is scored with four outcomes and two summary rates:

| Code | Meaning |
|------|---------|
| CBUG | Correctly Blocked Unsolvable Grid |
| IPUG | Incorrectly Processed Unsolvable Grid |
| WBSG | Wrongly Blocked Solvable Grid |
| CPSG | Correctly Processed Solvable Grid |
| SGR  | Solvable-decision rate = (CBUG + CPSG) / total |
| UGR  | Unsolvable-decision rate = (IPUG + WBSG) / total |

### Unified inference (gate + planner)

With `unified_framework: True`, the selected classifier gates every grid in the RAGate test
split before the Transformer rollout: grids predicted *No Route Available* are blocked, and only
those predicted *Route Available* are rolled out. Set `inference: False` when using this so the
standalone rollout does not also run.

## Results

Training and inference produce route visualizations and overview plots under `Results/`.

<!-- Add result images here, e.g.:
![Predicted vs. optimal route](docs/route_example.png)
![Gate comparison](docs/gate_comparison.png)
-->

## Project structure

```
PSTAR/
├── main.py                     # entry point; runs the stages enabled in config.py
├── config.py                   # central, fully-commented configuration
├── requirements.txt
├── PSTARModel/
│   ├── transformer.py          # PerceiverAR backbone (2D sinusoidal PE)
│   ├── direction_model.py      # U/D/R/L logits + loss wrapper
│   ├── sinusoidal_2dpe.py      # 2D positional encoding
│   ├── training.py             # autoregressive teacher-forcing training loop
│   ├── inference.py            # test-set rollout + checkpoint loading
│   ├── route_evaluation.py     # route metrics and overview plots
│   ├── stuck_masked_subtypes.py
│   └── visualization.py
├── RAGateModel/
│   ├── extract_features.py     # embeddings from the trained transformer
│   ├── lda_classifier.py
│   ├── mlp_classifier.py
│   ├── xgboost_classifier.py
│   ├── classifier_utils.py     # gate metrics (CBUG/IPUG/WBSG/CPSG, SGR/UGR)
│   └── compare_classifiers.py
└── shared/
    ├── dataloader.py           # streaming dataloader (pt / json / pickle)
    ├── unified_framework.py    # gate + rollout inference
    └── utils.py
```

## Related repository

* [`ASTAR_Dataset_Generation`](../ASTAR_Dataset_Generation) — generates the A\* grid datasets
  (solvable/unsolvable splits and the RAGate classifier dataset) consumed here.

## License

This project is released under the [MIT License](LICENSE).
