"""
config.py: Central configuration for the pathfinding transformer ML pipeline.
Optimized for goal-directed learning with hybrid masking strategy.

FEATURE TOGGLES
---------------
Features are numbered 0-7 and activated with the flags below.
feature_size is computed AUTOMATICALLY — do NOT set it manually.

  f0: cell_num        (always on)
  f1: x               (always on)
  f2: y               (always on)
  f3: state_status    (always on)
  f4: dx_to_goal      (feature_dx_dy toggle — adds f4 AND f5 together)
  f5: dy_to_goal      (part of feature_dx_dy toggle)
  f6: free_neighbours (feature_free_neighbours toggle)
  f7: is_interior     (feature_is_interior toggle)

CUDA DIAGNOSTICS
----------------
The pipeline prints a CUDA diagnostic block at startup (see main.py).
If you see "device=cpu", CUDA is not visible — check your driver/environment.
config["device"] resolves at import time:  "cuda" if torch.cuda.is_available() else "cpu"
"""

import torch

# ---------------------------------------------------------------------------
# SUBFOLDER PATH REGISTRATION
# Adds each subfolder to sys.path so all scripts can use flat imports
# (e.g. `from utils import ...`) regardless of which subfolder they live in.
# This must run before any other imports in the pipeline.
# ---------------------------------------------------------------------------
import sys as _sys
import os as _os
_project_root = _os.path.dirname(_os.path.abspath(__file__))
for _sub in ("PSTARModel", "RAGateModel", "shared"):
    _sub_path = _os.path.join(_project_root, _sub)
    if _sub_path not in _sys.path:
        _sys.path.insert(0, _sub_path)
del _sys, _os, _project_root, _sub, _sub_path



# ---------------------------------------------------------------------------
# FEATURE TOGGLE FLAGS  (the only knobs you need to turn)
# ---------------------------------------------------------------------------
_FEATURE_DX_DY          = False   # f4 + f5: dx_to_goal, dy_to_goal  (adds 2 features)
_FEATURE_FREE_NEIGHBOURS = False   # f6: normalised free-neighbour count [0,1]
_FEATURE_IS_INTERIOR     = False   # f7: 1=fully interior cell, 0=grid edge/corner

# ---------------------------------------------------------------------------
# AUTO-COMPUTED feature_size  — DO NOT EDIT
# ---------------------------------------------------------------------------
def _compute_feature_size(dx_dy, free_nbrs, interior):
    """Return the total feature vector length from the active toggles."""
    size = 4                        # f0-f3 always present
    if dx_dy:     size += 2         # f4 + f5
    if free_nbrs: size += 1         # f6
    if interior:  size += 1         # f7
    return size

_FEATURE_SIZE = _compute_feature_size(
    _FEATURE_DX_DY, _FEATURE_FREE_NEIGHBOURS, _FEATURE_IS_INTERIOR
)

# ---------------------------------------------------------------------------
# FEATURE INDEX MAP  (slot positions in the feature vector)
# ---------------------------------------------------------------------------
# Slots are assigned contiguously in activation order.
# Inactive features are not present — there are NO zero-filled gaps.
# Use these constants anywhere you need a specific feature by index.
_F_CELL_NUM    = 0
_F_X           = 1
_F_Y           = 2
_F_STATUS      = 3
_F_DX          = 4 if _FEATURE_DX_DY else None
_F_DY          = 5 if _FEATURE_DX_DY else None
_F_FREE_NBRS   = (4 + (2 if _FEATURE_DX_DY else 0))     if _FEATURE_FREE_NEIGHBOURS else None
_F_IS_INTERIOR = (4 + (2 if _FEATURE_DX_DY else 0)
                    + (1 if _FEATURE_FREE_NEIGHBOURS else 0)) if _FEATURE_IS_INTERIOR else None


config = {
    # ========================================================================
    # GRID AND DATA PARAMETERS
    # ========================================================================
    "grid_size": (10, 10),           # Grid dimensions (rows, cols)
    # feature_size is set automatically below — do not edit here
    "feature_size": _FEATURE_SIZE,
    "max_neighbours": 4,             # Max free neighbours per cell (4=cardinal)
    "one_indexed": True,             # Grid coordinates are 1-indexed
    "padding_value": -1,             # Value for padding in sequences

    # ========================================================================
    # FEATURE TOGGLE FLAGS  (mirrors the module-level flags above)
    # ========================================================================
    # Read from these in the rest of the pipeline (dataloader, direction_model, etc.)
    "feature_dx_dy":           _FEATURE_DX_DY,
    "feature_free_neighbours": _FEATURE_FREE_NEIGHBOURS,
    "feature_is_interior":     _FEATURE_IS_INTERIOR,

    # Feature index map — use these instead of hardcoded 4, 5, 6, 7
    "fidx_cell_num":    _F_CELL_NUM,
    "fidx_x":           _F_X,
    "fidx_y":           _F_Y,
    "fidx_status":      _F_STATUS,
    "fidx_dx":          _F_DX,           # None when feature_dx_dy=False
    "fidx_dy":          _F_DY,           # None when feature_dx_dy=False
    "fidx_free_nbrs":   _F_FREE_NBRS,    # None when feature_free_neighbours=False
    "fidx_is_interior": _F_IS_INTERIOR,  # None when feature_is_interior=False

    # ========================================================================
    # SEQUENCE LENGTHS
    # ========================================================================
    "context_matrix_length": 100,    # Length of grid context (all cells)
    "foundroute_matrix_length": 75,  # Max length of path sequence
    "target_matrix_length": 75,      # Max length of target directions
    "sequence_len": 175,             # Total sequence: context(100) + latent(75)

    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    "num_unique_tokens": 4,          # Direction tokens: U(0), D(1), R(2), L(3)
    "dim": 512,                      # Model embedding dimension
    "num_layers": 12,                # Number of transformer layers
    "heads": 8,                      # Number of attention heads
    "causal": True,                  # Use causal (autoregressive) attention
    "ff_mult": 4,                    # Feedforward dimension multiplier
    "ff_dropout": 0.1,               # Dropout rate in feedforward layers
    "attn_dropout": 0.1,             # Dropout rate in attention layers

    # ========================================================================
    # DIRECTION AND STATUS MAPPINGS
    # ========================================================================
    "reverse_direction_mapping": {
        0: "U",  # Up
        1: "D",  # Down
        2: "R",  # Right
        3: "L"   # Left
    },
    "status_start":    1,            # Grid cell status: start position
    "status_end":      2,            # Grid cell status: goal position
    "status_obstacle": 3,            # Grid cell status: obstacle/blocked
    "status_free":     4,            # Grid cell status: free/traversable
    "status_path":     5,            # Grid cell status: part of path

    # ========================================================================
    # TRAINING, Inference, and Unifined Pipeline CONFIGURATION
    # ========================================================================
    "pstar_training":            True,  # =True train the PSTAR model
    "inference":                 False,  # =True generting routes withour route calssifier
    "ragate_training": False,  # =True train the RAGate classifiers
    "unified_framework": False,  # =True Unifined pipeline 
    "batch": 128,
    "epochs": 200,
    "shuffle": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0,
    "pin_memory": True ,
    "file_format": "tensor_pt",      # "tensor_pt" | "json" | "pickle"
    "use_torch_compile": False,
    "ragate_embeddings_batch_size":  128,   # samples per forward pass during extraction

    # Training mode: autoregressive teacher forcing (full GT latent; the
    # Attention-2 causal mask prevents future-step leakage).
    "training_mode": "autoregressive",

    # ========================================================================
    # OPTIMIZATION PARAMETERS
    # ========================================================================
    "optimizer": "adam",
    "learning_rate": 2e-4,
    "gradient_clip_value": 1.0,
    "adam_params": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.00,
    },
    "momentum": 0.9,

    # ========================================================================
    # LOSS FUNCTION WEIGHTS
    # ========================================================================
    "ce_temperature": 1.0,
    "masking_value": float('-inf'),
    "goal_distance_weight":  0.10,
    "goal_proximity_weight": 0.00,
    "goal_obstacle_weight":  0.00,

    # ========================================================================
    # DATASET PATHS
    # ========================================================================
    "data_dir": "Datasets/PSTAR/",
    "train_file": "Datasets/PSTAR/train/train_sol.json",
    "val_file":   "Datasets/PSTAR/val/val_sol.json",
    "test_file":  "Datasets/PSTAR/test/test_sol.json",
    "train_file_pickle": "Datasets/PSTAR/train/train_sol.pkl",
    "val_file_pickle":   "Datasets/PSTAR/val/val_sol.pkl",
    "test_file_pickle":  "Datasets/PSTAR/test/test_sol.pkl",

    # ========================================================================
    # CHECKPOINTING AND RESULTS 
    # ========================================================================
    "checkpoint_dir": "Results/PSTAR/checkpoints/",
    "results_dir":    "Results/PSTAR/",
    "resume_from_checkpoint": "",    # set to "Results/PSTAR/checkpoints/last_model.pt" to resume from last training epoch
    "last_model_name":      "last_model.pt",
    "best_model_name":      "best_model.pt",
    "ce_best_model_name":   "ce_best_model.pt",
    "inference_model_name": "ce_best_model.pt",

    # ========================================================================
    # EVALUATION SETTINGS
    # ========================================================================
    "train_eval_every_n_epochs": 0,

    # ========================================================================
    # VISUALIZATION SETTINGS
    # ========================================================================
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

    "viz_show_dxdy_in_cells":           False,  # off: dx/dy not in feature set (4F experiment)

    "rollout_use_dataset_mask":         False,
    "dataset_integrity_check":          False,
    "dataset_integrity_sample_size":    5,
    "data_streaming":                   True,

    # =========================================================================
    # RANDOM SEED
    # =========================================================================
    "random_seed": 42,

    # =========================================================================
    # PHASE 1 — RAGATE CLASSIFIER — DATASET SPLITS  (produced by Phase 1 pipeline)
    # =========================================================================
    # Each .pt file is a list of sample dicts with keys:
    #   input_tensor, masking_tensor, target_tensor, context, found_routes, label
    # label: 1 = sol (solvable), 0 = nosol (no solution)
    "ragate_train_pt":   "Datasets/RAGate/train/RAGate_train.pt",
    "ragate_val_pt":     "Datasets/RAGate/val/RAGate_val.pt",
    "ragate_test_pt":    "Datasets/RAGate/test/RAGate_test.pt",

    # frozen the PSTAR chkpoint    
    # True = frozen (current behaviour) or False = unfrozen (new behaviour)
    "ragate_pstar_frozen": True,  

    # =========================================================================
    # PHASE 2 — RAGATE - GENERATING DATASET EMBEDDINGS
    # =========================================================================
    # Output .pt files from extract_features.py.
    # Keys present depend on active toggle flags below.
    #   "X"       — mean context embedding  [N, dim]  
    #   "X_start" — start cell embedding    [N, dim]  
    #   "X_goal"  — goal cell embedding     [N, dim]  
    #   "y"       — labels as torch.Tensor  [N]
    #   "labels"  — labels as plain list    [N]
    "ragate_embeddings_force_reextract": False,  # False = reuse existing (mode-tagged) embedding files; set True to force re-extraction after changing features or the checkpoint.
    "ragate_embeddings_train":  "Datasets/RAGate/extracted_features/embeddings_context_train.pt",
    "ragate_embeddings_val":    "Datasets/RAGate/extracted_features/embeddings_context_val.pt",
    "ragate_embeddings_test":   "Datasets/RAGate/extracted_features/embeddings_context_test.pt",


    # ── Embedding layer mode ──────────────────────────────────────────────────
    # False (default): extract context_mean from the last transformer layer only
    #                  → context_mean shape [B, dim]           e.g. [B, 512]
    # True:            extract context_mean from ALL num_layers transformer layers
    #                  and concatenate them along the feature dimension
    #                  → context_mean shape [B, dim * num_layers] e.g. [B, 6144]
    # Note: start_emb and goal_emb always use the last layer regardless of this flag.
    # Note: changing this flag requires force_reextract=True to regenerate embeddings.
    "ragate_embedding_all_layers":  True,

    # ── Embedding toggle flags — each is fully independent ───────────────────
    # Enabling one flag does NOT enable the others.
    # ragate_use_mean_emb must stay True for classifier scripts to find key "X".
    "ragate_use_mean_emb":    True,    # extract mean context embedding → "X"
    "ragate_use_start_emb":   True,   # extract start cell embedding   → "X_start" (last layer only)
    "ragate_use_goal_emb":    True,   # extract goal cell embedding    → "X_goal"  (last layer only)

    # =========================================================================
    # RAGATE CLASSIFIER — MODEL SELECTION AND THRESHOLD
    # =========================================================================
    # Two-outcome system:
    #   P(sol|emb) >= tau  →  "Route Available"    (sent to rollout)
    #   P(sol|emb) <  tau  →  "No Route Available" (blocked, no rollout)
    #
    # tau1_active = False disables the gate entirely (all grids proceed).
    "ragate_model":  "xgboost",  # "lda" | "mlp" | "xgboost"
    "ragate_model_tau1":         0.95,   # feasibility threshold
    "ragate_model_tau1_active":  True,   # False = skip gate, all grids proceed

    # ── Threshold sweep (config-driven; the sweep function reads this list) ────
    #    The deployment threshold (ragate_model_tau1) is added automatically.
    "ragate_sweep_values": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                               0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99],

    # ── Append-only gate results log (accumulates frozen + unfrozen runs) ─────
    #    NOT mode-tagged: one row per (mode x classifier x split).
    "ragate_gate_log": "Results/RAGate/log/gate_log.csv",
    "ragate_visualization_dir": "Results/RAGate/visualization/",

    # =========================================================================
    # RAGATE CLASSIFIER — LDA MODEL
    # =========================================================================
    "ragate_lda_model_path":   "Results/RAGate/checkpoints/RAGate_lda.pkl",
    "ragate_lda_priors":       [0.07, 0.93],
    "ragate_lda_eval_output":  "Results/RAGate/log/RAGate_lda_eval.json",
    "ragate_lda_sweep_output": "Results/RAGate/log/RAGate_lda_sweep.json",

    # =========================================================================
    # RAGATE CLASSIFIER — BINARY MODEL
    # =========================================================================
    "ragate_mlp_model_path":     "Results/RAGate/checkpoints/RAGate_mlp.pt",
    "ragate_mlp_lr":             3e-4,
    "ragate_mlp_epochs":         200,
    "ragate_mlp_dropout":        0.1,
    "ragate_mlp_weight_decay":   0,
    "ragate_mlp_batch_size":     128,
    # Hidden layer sizes for RAGateMLPNet.
    # Network is built dynamically from this list:
    #   [256, 64]        → A: 3-layer  512→256→64→1        (original)
    #   [256, 64, 32]    → B: 4-layer  512→256→64→32→1
    #   [256, 64, 32, 16]→ C: 5-layer  512→256→64→32→16→1
    # Each hidden size gets: Linear → GELU → Dropout
    # Final layer: Linear(last_hidden → 1), no activation (raw logit)
    "ragate_mlp_hidden_layers":  [512, 256, 64],
    # pos_weight: scales the loss on the positive class (solvable, label=1).
    # Values < 1.0 → conservative: model penalised less for missing solvables,
    #   more cautious about blocking — higher CBUG Rate, higher WBSG Rate.
    # Values > 1.0 → permissive: model penalised more for missing solvables,
    #   passes more grids through — lower WBSG Rate, lower CBUG Rate.
    # 1.0 = standard BCE (no asymmetry). Dataset is 1:1 balanced.
    # Recommended starting point: 0.5 for conservative blocking behaviour.
    "ragate_mlp_pos_weight":     0.3,
    "ragate_mlp_eval_output":    "Results/RAGate/log/RAGate_mlp_eval.json",
    "ragate_mlp_sweep_output":   "Results/RAGate/log/RAGate_mlp_sweep.json",
    "ragate_mlp_training_log":   "Results/RAGate/log/RAGate_mlp_training_log.csv",

    # =========================================================================
    # RAGATE CLASSIFIER — XGBOOST MODEL
    # =========================================================================
    "ragate_xgboost_model_path":    "Results/RAGate/checkpoints/RAGate_xgboost.pkl",
    "ragate_xgboost_n_estimators":  800,     # number of boosting rounds
    "ragate_xgboost_max_depth":     8,        # max tree depth (default=6)
    "ragate_xgboost_learning_rate":  0.05,    # step size shrinkage
    "ragate_xgboost_subsample":     0.9,      # row subsampling per tree
    "ragate_xgboost_colsample":     0.8,      # feature subsampling per tree
    "ragate_xgboost_eval_output":   "Results/RAGate/log/RAGate_xgboost_eval.json",
    "ragate_xgboost_sweep_output":  "Results/RAGate/log/RAGate_xgboost_sweep.json",

    # =========================================================================
    # RAGATE CLASSIFIER — SUCCESS CRITERIA  (shared by all three models)
    # =========================================================================
    # SGR = Solvable Grids Rate: (CBUG + CPSG) / total — overall correct decision rate
    # UGR = Unsolvable Grids Rate: (IPUG + WBSG) / total — overall wrong decision rate
    "ragate_success_recall_min":  0.95,   # minimum CBUG Rate required
    "ragate_success_fpr_max":     0.02,   # maximum WBSG Rate allowed

    # =========================================================================
    # RAGATE CLASSIFIER — COMPARISON OUTPUT
    # =========================================================================
    "ragate_comparison_output":  "Results/RAGate/log/RAGate_comparison.json",
    "ragate_inference_output":   "Results/RAGate/log/RAGate_inference_results.json",

    # =========================================================================
    # UNIFIED INFERENCE WITH CLASSIFIER
    # =========================================================================
    # When True, unified_framework() is called from main.py.
    # The classifier gates each sample before the transformer rollout.
    # Uses the RAGate test split (RAGate_test_pt) which contains
    # both solvable and unsolvable grids — required to evaluate the full gate.
    # Set inference=False when using this to avoid running inference_main() too.
    "inference_classifier_results_output": "Results/RAGate/log/unified_framework_results.json",

    # Visualization for unified inference with classifier
    # Blocked samples: show grid with classifier outcome label, no route
    # Proceed samples: show predicted route vs true route with outcome + probability
    "enable_unified_visualization":         True,
    "num_unified_visualizations_blocked":   10,   # max blocked samples to visualize
    "num_unified_visualizations_proceed":   10,   # max proceed samples to visualize
}

# ---------------------------------------------------------------------------
# Startup banner: print feature layout and CUDA status.
# Imported by main.py before any other module so the user sees this first.
# ---------------------------------------------------------------------------
def print_startup_diagnostics():
    """Print feature layout and CUDA status. Called once from main.py."""
    import torch as _torch

    cuda_available = _torch.cuda.is_available()
    device_str     = config["device"]

    print("\n" + "=" * 70)
    print("  STARTUP DIAGNOSTICS")
    print("=" * 70)

    # ── CUDA ──
    print(f"  PyTorch version : {_torch.__version__}")
    print(f"  CUDA available  : {cuda_available}")
    if cuda_available:
        print(f"  CUDA device     : {_torch.cuda.get_device_name(0)}")
        print(f"  CUDA count      : {_torch.cuda.device_count()}")
        vram = _torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM (device 0) : {vram:.1f} GB")
    else:
        print("  [WARNING] CUDA not found — training will run on CPU (slow).")
        print("  Check: nvidia-smi, CUDA toolkit version, torch+cu* install.")
    print(f"  Active device   : {device_str}")

    # ── Feature layout ──
    flags = {
        "feature_dx_dy":           config["feature_dx_dy"],
        "feature_free_neighbours": config["feature_free_neighbours"],
        "feature_is_interior":     config["feature_is_interior"],
    }
    feature_map = [
        ("f0", "cell_num",        True),
        ("f1", "x",               True),
        ("f2", "y",               True),
        ("f3", "state_status",    True),
        ("f4", "dx_to_goal",      config["feature_dx_dy"]),
        ("f5", "dy_to_goal",      config["feature_dx_dy"]),
        ("f6", "free_neighbours", config["feature_free_neighbours"]),
        ("f7", "is_interior",     config["feature_is_interior"]),
    ]
    print(f"\n  feature_size    : {config['feature_size']}")
    print("  Feature layout  :")
    for slot, (fname, fdesc, active) in enumerate(feature_map):
        if active:
            print(f"    [{slot}] {fname} ({fdesc})")
    print()
    for flag, val in flags.items():
        status = "ON " if val else "OFF"
        print(f"  {flag:30s}: {status}")
    print("=" * 70 + "\n")


# =============================================================================
# Mode-tag RAGate artifacts so frozen/unfrozen runs do not overwrite one
# another. The append-only gate_log.csv is intentionally NOT tagged.
# =============================================================================
_gc_mode = "frozen" if config.get("ragate_pstar_frozen", True) else "unfrozen"
import os as _os  # re-import: the module-level _os was del'd after sys.path setup
def _gc_tag_path(_path, _tag):
    _root, _ext = _os.path.splitext(_path)
    return f"{_root}_{_tag}{_ext}"
for _k in (
    "ragate_embeddings_train", "ragate_embeddings_val", "ragate_embeddings_test",
    "ragate_lda_model_path",    "ragate_lda_eval_output",    "ragate_lda_sweep_output",
    "ragate_mlp_model_path", "ragate_mlp_eval_output", "ragate_mlp_sweep_output",
    "ragate_xgboost_model_path","ragate_xgboost_eval_output","ragate_xgboost_sweep_output",
    "ragate_comparison_output", "ragate_inference_output",
):
    if config.get(_k):
        config[_k] = _gc_tag_path(config[_k], _gc_mode)
