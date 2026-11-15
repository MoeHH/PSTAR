"""PSTAR / RAGate configuration. Edit the EXPERIMENT block; everything else has working defaults."""
import os as _os
import sys as _sys
import torch

# Make the subfolders importable with flat imports (must run first).
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _sub in ("PSTARModel", "RAGateModel", "shared", "tools"):
    _p = _os.path.join(_ROOT, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sub, _p

# =============================================================================
# EXPERIMENT
# =============================================================================
_GRID_SIZE     = (10, 10)            # must match the A* generator
_LATENT_LENGTH = 75                  # route slots: 75 for 10x10, 150 for 20x20
_SPLIT_MODE    = "sample_disjoint"   # "sample_disjoint" | "template_disjoint"
_ARCHITECTURE  = "pstar"             # "pstar" | "perceiverar" (planner baseline, no RAGate)

# Optional input features (F0-F3 are always used)
_FEATURE_DX_DY           = False     # F4 + F5: dx, dy to goal
_FEATURE_FREE_NEIGHBOURS = False     # F6: free-neighbour count
_FEATURE_IS_INTERIOR     = False     # F7: interior-cell flag

_FEATURE_SIZE = 4 + 2 * _FEATURE_DX_DY + _FEATURE_FREE_NEIGHBOURS + _FEATURE_IS_INTERIOR
_F_DX = 4 if _FEATURE_DX_DY else None
_F_DY = 5 if _FEATURE_DX_DY else None
_F_FREE_NBRS = (4 + 2 * _FEATURE_DX_DY) if _FEATURE_FREE_NEIGHBOURS else None
_F_IS_INTERIOR = (4 + 2 * _FEATURE_DX_DY + _FEATURE_FREE_NEIGHBOURS) if _FEATURE_IS_INTERIOR else None

config = {
    # ---- Pipeline stages (run in this order by main.py) ---------------------
    "pstar_training":    True,
    "inference":         True,       # planner only, on the PSTAR test set
    "ragate_training":   True,       # LDA, MLP and XGBoost gates
    "unified_framework": True,       # gate + planner on the RAGate test set
    "timing_benchmark":  False,

    # ---- Reproducibility ----------------------------------------------------
    "random_seed":   42,             # 32, 42, and 52 
    "deterministic": True,           # deterministic GPU kernels (slower)
    "resume_from_checkpoint": "",    # path to <arch>_last_model.pt to resume

    # ---- Experiment (from the block above) ----------------------------------
    "grid_size":    _GRID_SIZE,
    "architecture": _ARCHITECTURE,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",

    # ---- Input features -----------------------------------------------------
    "feature_size":            _FEATURE_SIZE,
    "feature_dx_dy":           _FEATURE_DX_DY,
    "feature_free_neighbours": _FEATURE_FREE_NEIGHBOURS,
    "feature_is_interior":     _FEATURE_IS_INTERIOR,
    "fidx_cell_num": 0, "fidx_x": 1, "fidx_y": 2, "fidx_status": 3,
    "fidx_dx": _F_DX, "fidx_dy": _F_DY,
    "fidx_free_nbrs": _F_FREE_NBRS, "fidx_is_interior": _F_IS_INTERIOR,

    # ---- Sequence lengths (derived below; leave None) -----------------------
    "context_matrix_length":    None,
    "foundroute_matrix_length": _LATENT_LENGTH,
    "target_matrix_length":     None,
    "sequence_len":             None,

    # ---- Model --------------------------------------------------------------
    "dim":          512,
    "num_layers":   12,
    "heads":        8,
    "ff_mult":      4,
    "ff_dropout":   0.1,
    "attn_dropout": 0.1,
    "causal":       True,
    "num_unique_tokens": 4,          # U, D, R, L

    # ---- Training -----------------------------------------------------------
    "epochs":        200,
    "batch":         128,
    "optimizer":     "adam",
    "learning_rate": 2e-4,
    "adam_params":   {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
    "momentum":      0.9,
    "gradient_clip_value": 1.0,
    "training_mode": "autoregressive",
    "shuffle":       True,
    "num_workers":   0,
    "pin_memory":    True,
    "use_torch_compile": False,
    "train_eval_every_n_epochs": 0,

    # ---- Loss ---------------------------------------------------------------
    "goal_distance_weight":  0.10,   # w_delta
    "goal_proximity_weight": 0.00,
    "goal_obstacle_weight":  0.00,
    "ce_temperature":        1.0,
    "masking_value":         float("-inf"),

    # ---- Grid encoding ------------------------------------------------------
    "max_neighbours": 4,
    "one_indexed":    True,
    "padding_value":  -1,
    "reverse_direction_mapping": {0: "U", 1: "D", 2: "R", 3: "L"},
    "status_start": 1, "status_end": 2, "status_obstacle": 3, "status_free": 4, "status_path": 5,
    "template_groups": "1-35,36-100,101-150",

    # ---- Data and results (namespaced by grid and split below) --------------
    "file_format":       "tensor_pt",          # "tensor_pt" | "json" | "pickle"
    "data_streaming":    True,
    "data_dir":          "Datasets/PSTAR/",
    "train_file":        "Datasets/PSTAR/train/train_sol.json",
    "val_file":          "Datasets/PSTAR/val/val_sol.json",
    "test_file":         "Datasets/PSTAR/test/test_sol.json",
    "train_file_pickle": "Datasets/PSTAR/train/train_sol.pkl",
    "val_file_pickle":   "Datasets/PSTAR/val/val_sol.pkl",
    "test_file_pickle":  "Datasets/PSTAR/test/test_sol.pkl",
    "checkpoint_dir":    "Results/PSTAR/checkpoints/",
    "results_dir":       "Results/PSTAR/",
    "last_model_name":      "last_model.pt",
    "best_model_name":      "best_model.pt",
    "ce_best_model_name":   "ce_best_model.pt",
    "inference_model_name": "ce_best_model.pt",

    # ---- RAGate: data and embeddings ----------------------------------------
    "ragate_train_pt": "Datasets/RAGate/train/RAGate_train.pt",
    "ragate_val_pt":   "Datasets/RAGate/val/RAGate_val.pt",
    "ragate_test_pt":  "Datasets/RAGate/test/RAGate_test.pt",
    "ragate_pstar_frozen":               True,
    "ragate_embeddings_force_reextract": True,
    "ragate_embeddings_batch_size":      128,
    "ragate_embedding_all_layers":       True,   # context mean of all layers
    "ragate_use_mean_emb":               True,
    "ragate_use_start_emb":              True,
    "ragate_use_goal_emb":               True,
    "ragate_embeddings_train": "Datasets/RAGate/extracted_features/embeddings_context_train.pt",
    "ragate_embeddings_val":   "Datasets/RAGate/extracted_features/embeddings_context_val.pt",
    "ragate_embeddings_test":  "Datasets/RAGate/extracted_features/embeddings_context_test.pt",

    # ---- RAGate: gate and threshold -----------------------------------------
    "ragate_model":             "xgboost",       # gate used by the unified pipeline
    "ragate_model_tau1":        0.95,            # admit a map if P(route available) >= tau
    "ragate_model_tau1_active": True,
    "ragate_sweep_values": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                            0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99],
    "ragate_success_cbug_rate_min": 0.95,
    "ragate_success_wbsg_rate_max": 0.02,

    # ---- RAGate: classifiers ------------------------------------------------
    "ragate_lda_priors":            [0.07, 0.93],
    "ragate_mlp_hidden_layers":     [512, 256, 64],
    "ragate_mlp_lr":                3e-4,
    "ragate_mlp_epochs":            200,
    "ragate_mlp_batch_size":        128,
    "ragate_mlp_dropout":           0.1,
    "ragate_mlp_weight_decay":      0,
    "ragate_mlp_pos_weight":        0.3,
    "ragate_xgboost_n_estimators":  800,
    "ragate_xgboost_max_depth":     8,
    "ragate_xgboost_learning_rate": 0.05,
    "ragate_xgboost_subsample":     0.9,
    "ragate_xgboost_colsample":     0.8,

    # ---- RAGate: joint fine-tuning (optional, MLP head only) ----------------
    "ragate_joint_finetune":            False,
    "ragate_joint_finetune_epochs":     20,
    "ragate_joint_finetune_lr":         1e-5,
    "ragate_joint_finetune_checkpoint": "Results/PSTAR/checkpoints/ce_best_model_finetuned.pt",
    "ragate_step0_eval_head":           False,
    "ragate_step0_eval_output":         "Results/RAGate/log/RAGate_step0_eval.json",

    # ---- RAGate: output files -----------------------------------------------
    "ragate_lda_model_path":       "Results/RAGate/checkpoints/RAGate_lda.pkl",
    "ragate_lda_eval_output":      "Results/RAGate/log/RAGate_lda_eval.json",
    "ragate_lda_sweep_output":     "Results/RAGate/log/RAGate_lda_sweep.json",
    "ragate_mlp_model_path":       "Results/RAGate/checkpoints/RAGate_mlp.pt",
    "ragate_mlp_eval_output":      "Results/RAGate/log/RAGate_mlp_eval.json",
    "ragate_mlp_sweep_output":     "Results/RAGate/log/RAGate_mlp_sweep.json",
    "ragate_mlp_training_log":     "Results/RAGate/log/RAGate_mlp_training_log.csv",
    "ragate_xgboost_model_path":   "Results/RAGate/checkpoints/RAGate_xgboost.pkl",
    "ragate_xgboost_eval_output":  "Results/RAGate/log/RAGate_xgboost_eval.json",
    "ragate_xgboost_sweep_output": "Results/RAGate/log/RAGate_xgboost_sweep.json",
    "ragate_comparison_output":    "Results/RAGate/log/RAGate_comparison.json",
    "ragate_inference_output":     "Results/RAGate/log/RAGate_inference_results.json",
    "ragate_gate_log":             "Results/RAGate/log/gate_log.csv",
    "ragate_visualization_dir":    "Results/RAGate/visualization/",
    "inference_classifier_results_output": "Results/RAGate/log/unified_framework_results.json",

    # ---- Reports and visualization ------------------------------------------
    "write_run_report":                 True,
    "save_visualizations":              True,
    "show_visualizations":              False,
    "plot_model_overview_performance":  True,
    "show_predicted_route_in_training": True,
    "enable_training_visualization":    False,
    "num_train_visualizations":         5,
    "enable_validation_visualization":  False,
    "num_val_visualizations":           20,
    "enable_inference_visualization":   True,
    "num_inference_visualizations":     20,
    "enable_unified_visualization":     True,
    "num_unified_visualizations_blocked": 20,
    "num_unified_visualizations_proceed": 20,
    "viz_show_dxdy_in_cells":           False,
    "rollout_use_dataset_mask":         False,
    "dataset_integrity_check":          False,
    "dataset_integrity_sample_size":    5,

    # ---- Timing benchmark (optional) ----------------------------------------
    "timing_warmup_runs":     10,
    "timing_runs":            50,
    "timing_samples_per_bin": 20,
    "timing_percentiles":     [10, 50, 99],
    "timing_bin_names":       ["short", "medium", "long"],
    "timing_include_gate":    True,
    "timing_results_dir":     "Results/PSTAR/timing/",
}


def print_startup_diagnostics():
    """Print CUDA status and the active feature layout (called by main.py)."""
    print("\n" + "=" * 70 + "\n  STARTUP DIAGNOSTICS\n" + "=" * 70)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device     : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA count      : {torch.cuda.device_count()}")
        print(f"  VRAM (device 0) : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  [WARNING] CUDA not found - training will run on CPU (slow).")
    print(f"  Active device   : {config['device']}")
    layout = [("f0", "cell_num", True), ("f1", "x", True), ("f2", "y", True),
              ("f3", "state_status", True),
              ("f4", "dx_to_goal", config["feature_dx_dy"]), ("f5", "dy_to_goal", config["feature_dx_dy"]),
              ("f6", "free_neighbours", config["feature_free_neighbours"]),
              ("f7", "is_interior", config["feature_is_interior"])]
    print(f"\n  feature_size    : {config['feature_size']}\n  Feature layout  :")
    for slot, (name, desc, _) in enumerate([f for f in layout if f[2]]):
        print(f"    [{slot}] {name} ({desc})")
    print("=" * 70 + "\n")


# =============================================================================
# DERIVED SETTINGS - do not edit below this line
# =============================================================================
def _derive_or_check(key, derived):
    given = config.get(key)
    if given is None:
        config[key] = derived
    elif int(given) != int(derived):
        raise ValueError(f"config: {key}={given} contradicts grid_size / foundroute_matrix_length "
                         f"(expected {derived}). Leave it None.")

_n_rows, _n_cols = (int(v) for v in config["grid_size"])
_derive_or_check("context_matrix_length", _n_rows * _n_cols)
_derive_or_check("target_matrix_length",  _LATENT_LENGTH)
_derive_or_check("sequence_len",          _n_rows * _n_cols + _LATENT_LENGTH)

# Datasets/<RxC>_<split>/...  and  Results/{PSTAR|PerceiverAR|RAGate}_<N>_<split>/...
config["grid_tag"]    = f"{_n_rows}x{_n_cols}"
config["split_mode"]  = _SPLIT_MODE
config["dataset_tag"] = f"{config['grid_tag']}_{_SPLIT_MODE}"
_res_tag = (str(_n_rows) if _n_rows == _n_cols else config["grid_tag"]) + f"_{_SPLIT_MODE}"

def _namespace(path):
    if not isinstance(path, str):
        return path
    p = path.replace("\\", "/")
    if p.startswith("Datasets/") and not p.startswith(f"Datasets/{config['dataset_tag']}/"):
        return f"Datasets/{config['dataset_tag']}/" + p[len("Datasets/"):]
    for root in ("PSTAR", "RAGate"):
        prefix = f"Results/{root}/"
        if p.startswith(prefix) or p == prefix.rstrip("/"):
            out_root = "PerceiverAR" if (root == "PSTAR" and _ARCHITECTURE == "perceiverar") else root
            return f"Results/{out_root}_{_res_tag}/" + p[len(prefix):]
    return path

for _k, _v in list(config.items()):
    if isinstance(_v, str):
        config[_k] = _namespace(_v)

# Tag RAGate artifacts with the backbone mode so frozen/unfrozen runs never overwrite each other.
_mode = "frozen" if config.get("ragate_pstar_frozen", True) else "unfrozen"
for _k in ("ragate_embeddings_train", "ragate_embeddings_val", "ragate_embeddings_test",
           "ragate_lda_model_path", "ragate_lda_eval_output", "ragate_lda_sweep_output",
           "ragate_mlp_model_path", "ragate_mlp_eval_output", "ragate_mlp_sweep_output",
           "ragate_xgboost_model_path", "ragate_xgboost_eval_output", "ragate_xgboost_sweep_output",
           "ragate_comparison_output", "ragate_inference_output"):
    if config.get(_k):
        _root, _ext = _os.path.splitext(config[_k])
        config[_k] = f"{_root}_{_mode}{_ext}"

# Prefix checkpoint names with the architecture: pstar_ce_best_model.pt, perceiverar_...
if _ARCHITECTURE not in ("pstar", "perceiverar"):
    raise ValueError(f"config: _ARCHITECTURE={_ARCHITECTURE!r} - use 'pstar' or 'perceiverar'.")

def _arch_prefixed(name):
    base = _os.path.basename(name)
    if base.startswith(f"{_ARCHITECTURE}_"):
        return name
    d = _os.path.dirname(name)
    return _os.path.join(d, f"{_ARCHITECTURE}_{base}") if d else f"{_ARCHITECTURE}_{base}"

for _k in ("last_model_name", "best_model_name", "ce_best_model_name", "inference_model_name",
           "ragate_joint_finetune_checkpoint"):
    config[_k] = _arch_prefixed(config[_k])

# Perceiver AR is a planner-only baseline: the RAGate and the unified pipeline need PSTAR.
if _ARCHITECTURE == "perceiverar":
    _blocked = [k for k in ("ragate_training", "unified_framework", "ragate_joint_finetune") if config.get(k)]
    if config.get("timing_benchmark") and config.get("timing_include_gate", True):
        _blocked.append("timing_benchmark with timing_include_gate")
    if _blocked:
        raise ValueError("config: architecture='perceiverar' is a planner-only baseline - set these to False: "
                         + ", ".join(_blocked))

del _k, _v, _mode, _root, _ext
