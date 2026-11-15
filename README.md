This work was submitted to the AI (MDPI) Special Issue, Integrating Large Language Models into Robotic Autonomy (https://www.mdpi.com/journal/ai/special_issues/26397176N2) and is under review.

# PSTAR - A Unified One-Shot Autoregressive Model for Path Planning

PSTAR generates grid routes with a transformer in a single autoregressive rollout, without graph search.
An optional route-availability gate (RAGate) rejects unsolvable maps before decoding.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.4 (CPU works but is slow)

```bash
pip install -r requirements.txt
```

## 1. Get the dataset

Generate it with [ASTAR_Dataset_Generation](https://github.com/MoeHH/ASTAR_Dataset_Generation),
then copy its `Datasets/` folder into this folder:

```
PSTAR/
├── Datasets/10x10_sample_disjoint/...
├── config.py
└── main.py
```

## 2. Run

```bash
python main.py
```

With the default `config.py`, this runs the full pipeline in order:

1. trains PSTAR (200 epochs);
2. evaluates the planner on the test set;
3. trains and evaluates the RAGate classifiers (LDA, MLP, XGBoost);
4. evaluates the unified pipeline (gate + planner).

Each step can be switched on or off at the top of `config`:
`pstar_training`, `inference`, `ragate_training`, `unified_framework`.

## 3. Results

```
Results/
├── PSTAR_10_sample_disjoint/    # checkpoints, test results, run report
└── RAGate_10_sample_disjoint/   # gate models, metrics, unified-pipeline results
```

The run report (`report/pstar_report.md`) summarizes all results of the run.

## Common settings (top of `config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `_SPLIT_MODE` | `"sample_disjoint"` | Must match the generated dataset (`"template_disjoint"` for unseen templates) |
| `_ARCHITECTURE` | `"pstar"` | `"perceiverar"` trains the Perceiver AR baseline (planner only; set the RAGate and unified steps to `False`) |
| `random_seed` | `42` | Training seed |
| `deterministic` | `True` | Deterministic GPU kernels (same seed gives the same model) |
| `ragate_model_tau1` | `0.95` | Gate threshold: a map is admitted if P(route available) ≥ τ |
| `resume_from_checkpoint` | `""` | Path to `pstar_last_model.pt` to resume an interrupted run |

## Citation

If you use this code, please cite the PSTAR paper.
