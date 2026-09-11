import os
import csv
import numpy as np
from datetime import datetime

# =============================================================================
# _load_embeddings
# =============================================================================

def _load_embeddings(path):
    """
    Load a .pt embedding file.

    Returns (X_numpy, y_numpy, labels_list) where X is:
      - data["X_combined"]  [N, combined_dim]  if present (mean+start+goal)
      - data["X"]           [N, dim]            otherwise  (mean only)

    This allows all classifiers to work transparently with both
    512-dim (mean only) and 7168-dim (all-layers + start + goal) embeddings
    without any config changes.
    """
    import torch as _torch

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            f"Run extract_features.py first."
        )
    data = _torch.load(path, weights_only=False)

    if "X_combined" in data:
        X = data["X_combined"].numpy()
    elif "X" in data:
        X = data["X"].numpy()
    else:
        raise KeyError(
            f"Neither 'X_combined' nor 'X' found in {path}. "
            f"Ensure ragate_use_mean_emb=True when extracting embeddings."
        )

    y      = data["y"].numpy()
    labels = data.get("labels", y.tolist())
    return X, y, labels

# =============================================================================
# _apply_thresholds
# =============================================================================

def _apply_thresholds(probs, tau, tau1_active):

    probs      = np.asarray(probs)
    pred_nosol = np.zeros(len(probs), dtype=int)
    pred_sol   = np.ones(len(probs),  dtype=int)
    outcomes   = []

    for i, p in enumerate(probs):
        if tau1_active and p < tau:
            outcomes.append("No Route Available")
            pred_nosol[i] = 1
            pred_sol[i]   = 0
        else:
            outcomes.append("Route Available")

    return outcomes, pred_nosol, pred_sol


# =============================================================================
# _compute_metrics
# =============================================================================

def _compute_metrics(y_true, probs, pred_nosol, pred_sol):

    y_true = np.asarray(y_true, dtype=int)
    total  = int(len(y_true))

    nosol_mask  = (y_true == 0)
    sol_mask    = (y_true == 1)
    total_nosol = int(nosol_mask.sum())
    total_sol   = int(sol_mask.sum())

    # ── Four outcome counts ───────────────────────────────────────────────────
    CBUG = int((pred_nosol[nosol_mask] == 1).sum())  # Correctly Blocked Unsolvable Grid
    IPUG = int((pred_nosol[nosol_mask] == 0).sum())  # Incorrectly Processed Unsolvable Grid
    WBSG = int((pred_nosol[sol_mask]   == 1).sum())  # Wrongly Blocked Solvable Grid
    CPSG = int((pred_sol[sol_mask]     == 1).sum())  # Correctly Processed Solvable Grid

    # ── Metrics ───────────────────────────────────────────────────────────────
    cbug_rate = CBUG / max(CBUG + IPUG, 1)
    cpsg_rate = CPSG / max(total_sol,   1)
    wbsg_rate = WBSG / max(total_sol,   1)
    sgr_rate  = (CBUG + CPSG) / max(total, 1)
    ugr_rate  = (IPUG + WBSG) / max(total, 1)

    return {
        "total_samples":         total,
        "total_solvable":        total_sol,
        "total_unsolvable":      total_nosol,
        "CBUG":                  CBUG,
        "IPUG":                  IPUG,
        "WBSG":                  WBSG,
        "CPSG":                  CPSG,
        "cbug_rate":             round(cbug_rate, 4),
        "cpsg_rate":             round(cpsg_rate, 4),
        "wbsg_rate":             round(wbsg_rate, 4),
        "solvable_grids_rate":   round(sgr_rate,  4),
        "unsolvable_grids_rate": round(ugr_rate,  4),
    }


# =============================================================================
# _threshold_sweep
# =============================================================================

def _threshold_sweep(y_true, probs, tau1_active=True, sweep_values=None):

    if sweep_values is None:
        # Config-driven: read the sweep grid from config["ragate_sweep_values"]
        # and always include the deployment threshold ragate_model_tau1.
        try:
            from config import config as _cfg
            sweep_values = list(_cfg.get("ragate_sweep_values") or [])
            _active = _cfg.get("ragate_model_tau1")
            if _active is not None:
                sweep_values.append(float(_active))
        except Exception:
            sweep_values = []
        if not sweep_values:
            sweep_values = [0.30, 0.35, 0.40, 0.45, 0.50,
                            0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                            0.85, 0.90, 0.95]
        sweep_values = sorted({round(float(v), 4) for v in sweep_values})

    y_true = np.asarray(y_true, dtype=int)
    rows   = []

    for tau1_val in sweep_values:
        _, pred_nosol, pred_sol = _apply_thresholds(
            probs, tau1_val, tau1_active=True
        )
        metrics = _compute_metrics(y_true, probs, pred_nosol, pred_sol)
        rows.append({
            "tau":                  tau1_val,
            "cbug_rate":             metrics["cbug_rate"],
            "wbsg_rate":             metrics["wbsg_rate"],
            "solvable_grids_rate":   metrics["solvable_grids_rate"],
            "unsolvable_grids_rate": metrics["unsolvable_grids_rate"],
        })

    return rows


# =============================================================================
# gate_log.csv — append-only, mode-tagged results log (frozen + unfrozen)
# =============================================================================

def _append_csv_row(csv_path, row):

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_header = list(row.keys())
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_header)
            w.writeheader(); w.writerow(row)
        return
    with open(csv_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        old_rows = list(rd)
    if header == new_header:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)
        return
    union = list(header)
    for k in new_header:
        if k not in union:
            union.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=union)
        w.writeheader()
        for r0 in old_rows:
            w.writerow({k: r0.get(k, "") for k in union})
        w.writerow({k: row.get(k, "") for k in union})


def append_gate_log(classifier, split, metrics, tau, tau1_active=True,
                    model_size_kb=None, embedding_dim=None, csv_path=None):

    try:
        from config import config as _cfg
    except Exception:
        _cfg = {}
    mode = "frozen" if _cfg.get("ragate_pstar_frozen", True) else "unfrozen"
    if csv_path is None:
        csv_path = _cfg.get("ragate_gate_log", "results/gate_log.csv")
    row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode":            mode,
        "classifier":      classifier,
        "split":           split,
        "tau":            round(float(tau), 4),
        "tau1_active":     bool(tau1_active),
        "embedding_dim":   int(embedding_dim) if embedding_dim is not None else "",
        "N":               metrics.get("total_samples", ""),
        "solvable":        metrics.get("total_solvable", ""),
        "unsolvable":      metrics.get("total_unsolvable", ""),
        "CBUG":            metrics.get("CBUG", ""),
        "IPUG":            metrics.get("IPUG", ""),
        "WBSG":            metrics.get("WBSG", ""),
        "CPSG":            metrics.get("CPSG", ""),
        "cbug_rate":       metrics.get("cbug_rate", ""),
        "cpsg_rate":       metrics.get("cpsg_rate", ""),
        "wbsg_rate":       metrics.get("wbsg_rate", ""),
        "SGR":             metrics.get("solvable_grids_rate", ""),
        "UGR":             metrics.get("unsolvable_grids_rate", ""),
        "inference_ms_per_sample": metrics.get("inference_ms_per_sample", ""),
        "model_size_kb":   model_size_kb if model_size_kb is not None else "",
    }
    _append_csv_row(csv_path, row)
    return csv_path
