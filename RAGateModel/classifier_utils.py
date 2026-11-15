import os
import csv
import numpy as np
from datetime import datetime

def _load_embeddings(path):
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

def _safe_div(a, b):
    return (a / b) if b else None


def gate_metrics(TN, FP, FN, TP,
                 CBUG=None, IPUG=None, WBSG=None, CPSG=None):
    if CBUG is not None: TN = CBUG
    if IPUG is not None: FP = IPUG
    if WBSG is not None: FN = WBSG
    if CPSG is not None: TP = CPSG

    n_unsol, n_sol = TN + FP, TP + FN
    n = n_unsol + n_sol
    recall      = _safe_div(TP, n_sol)        # TP / (TP + FN)
    specificity = _safe_div(TN, n_unsol)      # TN / (TN + FP)
    mcc_den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    r = lambda v: None if v is None else round(float(v), 6)
    return {
        # Standard names
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "recall":              r(recall),
        "specificity":         r(specificity),
        "fn_rate":             r(_safe_div(FN, n_sol)),
        "accuracy":            r(_safe_div(TP + TN, n)),      # RGSR
        "error_rate":          r(_safe_div(FP + FN, n)),      # RGFR
        "precision":           r(_safe_div(TP, TP + FP)),
        "f1":                  r(_safe_div(2 * TP, 2 * TP + FP + FN)),
        "balanced_accuracy":   r(0.5 * (recall + specificity)
                                 if recall is not None and specificity is not None else None),
        "mcc":                 r(_safe_div(TP * TN - FP * FN, mcc_den)),
        # Legacy aliases (kept for backward compatibility with CSV columns etc.)
        "CBUG": TN, "IPUG": FP, "WBSG": FN, "CPSG": TP,
        "cbug_rate":             r(specificity),
        "cpsg_rate":             r(recall),
        "wbsg_rate":             r(_safe_div(FN, n_sol)),
        "solvable_grids_rate":   r(_safe_div(TP + TN, n)),
        "unsolvable_grids_rate": r(_safe_div(FP + FN, n)),
    }

def fmt_metric(v, percent=False, decimals=None):
    if v is None:
        return "n/a"
    if percent:
        return f"{100.0 * v:.{2 if decimals is None else decimals}f}%"
    return f"{v:.{4 if decimals is None else decimals}f}"

def fmt_pct(v, decimals=2):
    return fmt_metric(v, percent=True, decimals=decimals)

def print_gate_report(title, m, ms_per_sample=None, indent="  ", percent=False):
    f = lambda v: fmt_metric(v, percent=percent)
    print(f"\n{title}")
    print(f"{indent}Outcome Counts (solvable = positive class):")
    print(f"{indent}  True  Positive — solvable,   admitted  (TP): {m['TP']:,}")
    print(f"{indent}  True  Negative — unsolvable, blocked   (TN): {m['TN']:,}")
    print(f"{indent}  False Positive — unsolvable, admitted  (FP): {m['FP']:,}")
    print(f"{indent}  False Negative — solvable,   blocked   (FN): {m['FN']:,}")
    print(f"{indent}Metrics:")
    print(f"{indent}  Recall (sensitivity) TP/(TP+FN)             : {f(m['recall'])}")
    print(f"{indent}  Specificity          TN/(TN+FP)             : {f(m['specificity'])}")
    print(f"{indent}  FN Rate (miss rate)  FN/(TP+FN)             : {f(m['fn_rate'])}  <- primary gate")
    print(f"{indent}  Accuracy RGSR        (TP+TN)/N              : {f(m['accuracy'])}  <- overall correct")
    print(f"{indent}  Error Rate RGFR      (FP+FN)/N              : {f(m['error_rate'])}")
    print(f"{indent}  Precision            TP/(TP+FP)             : {f(m['precision'])}")
    print(f"{indent}  F1                   2TP/(2TP+FP+FN)        : {f(m['f1'])}")
    print(f"{indent}  Balanced accuracy    (Recall+Specificity)/2 : {f(m['balanced_accuracy'])}")
    mcc = m.get("mcc")
    print(f"{indent}  MCC                                         : "
          f"{'n/a' if mcc is None else f'{mcc:.4f}'}")
    if ms_per_sample is not None:
        print(f"{indent}  Inference: {ms_per_sample:.4f} ms/sample")

def _compute_metrics(y_true, probs, pred_nosol, pred_sol):
    y_true = np.asarray(y_true, dtype=int)
    total  = int(len(y_true))
    nosol_mask  = (y_true == 0)
    sol_mask    = (y_true == 1)
    total_nosol = int(nosol_mask.sum())
    total_sol   = int(sol_mask.sum())
    TN = int((pred_nosol[nosol_mask] == 1).sum())  # True  Negative: unsolvable, blocked
    FP = int((pred_nosol[nosol_mask] == 0).sum())  # False Positive: unsolvable, admitted
    FN = int((pred_nosol[sol_mask]   == 1).sum())  # False Negative: solvable, blocked
    TP = int((pred_sol[sol_mask]     == 1).sum())  # True  Positive: solvable, admitted
    recall      = TP / max(TP + FN, 1)   # sensitivity
    specificity = TN / max(TN + FP, 1)
    fn_rate     = FN / max(TP + FN, 1)
    accuracy    = (TP + TN) / max(total, 1)   # RGSR
    error_rate  = (FP + FN) / max(total, 1)   # RGFR
    metrics = {
        "total_samples":         total,
        "total_solvable":        total_sol,
        "total_unsolvable":      total_nosol,
        # Standard names
        "TP":                    TP,
        "TN":                    TN,
        "FP":                    FP,
        "FN":                    FN,
        "recall":                round(recall,      4),
        "specificity":           round(specificity, 4),
        "fn_rate":               round(fn_rate,     4),
        "accuracy":              round(accuracy,    4),
        "error_rate":            round(error_rate,  4),
        # Legacy aliases
        "CBUG": TN, "IPUG": FP, "WBSG": FN, "CPSG": TP,
        "cbug_rate":             round(specificity, 4),
        "cpsg_rate":             round(recall,      4),
        "wbsg_rate":             round(fn_rate,     4),
        "solvable_grids_rate":   round(accuracy,    4),
        "unsolvable_grids_rate": round(error_rate,  4),
    }
    # Added metrics: Precision, F1, balanced accuracy, MCC
    added = gate_metrics(TN=TN, FP=FP, FN=FN, TP=TP)
    for k in ("precision", "f1", "balanced_accuracy", "mcc"):
        metrics[k] = added[k]
    return metrics

def _threshold_sweep(y_true, probs, tau1_active=True, sweep_values=None):

    if sweep_values is None:
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
            "tau":              tau1_val,
            "recall":           metrics["recall"],
            "specificity":      metrics["specificity"],
            "fn_rate":          metrics["fn_rate"],
            "accuracy":         metrics["accuracy"],
            "error_rate":       metrics["error_rate"],
            "precision":        metrics["precision"],
            "f1":               metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "mcc":              metrics["mcc"],
        })

    return rows

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
        "TP":              metrics.get("TP", ""),
        "TN":              metrics.get("TN", ""),
        "FP":              metrics.get("FP", ""),
        "FN":              metrics.get("FN", ""),
        "recall":          metrics.get("recall", ""),
        "specificity":     metrics.get("specificity", ""),
        "fn_rate":         metrics.get("fn_rate", ""),
        "RGSR":            metrics.get("accuracy", ""),
        "RGFR":            metrics.get("error_rate", ""),
        "precision":       metrics.get("precision", ""),
        "f1":              metrics.get("f1", ""),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "mcc":             metrics.get("mcc", ""),
        "inference_ms_per_sample": metrics.get("inference_ms_per_sample", ""),
        "model_size_kb":   model_size_kb if model_size_kb is not None else "",
    }
    _append_csv_row(csv_path, row)
    return csv_path
