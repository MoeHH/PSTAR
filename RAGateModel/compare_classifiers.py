
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config

def _load_eval_json(path, model_label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[COMPARE] {model_label} eval file not found: {path}\n"
            f"          Run the corresponding --evaluate step first."
        )
    with open(path) as f:
        return json.load(f)

def _fmt(value, decimals=4):
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)

def _fmt_pct(value):
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value * 100:.2f}%"
    return str(value)

def _shortfall_str(model_label, specificity, fn_rate, recall_min, fpr_max):
    parts = []
    if specificity < recall_min:
        parts.append(f"Specificity={specificity*100:.2f}% < {recall_min*100:.0f}%")
    if fn_rate > fpr_max:
        parts.append(f"FN Rate={fn_rate*100:.2f}% > {fpr_max*100:.0f}%")
    return f"{model_label}: {', '.join(parts)}"

def main():
    recall_min = float(config.get("ragate_success_cbug_rate_min",
                                  config.get("ragate_success_recall_min", 0.95)))
    fpr_max    = float(config.get("ragate_success_wbsg_rate_max",
                                  config.get("ragate_success_fpr_max", 0.02)))
    lda_eval    = _load_eval_json(config["ragate_lda_eval_output"],      "LDA")
    nn_eval     = _load_eval_json(config["ragate_mlp_eval_output"],   "MLP")
    xgb_eval    = _load_eval_json(config["ragate_xgboost_eval_output"],  "XGBoost")
    lda_test    = lda_eval.get("test",    {})
    nn_test     = nn_eval.get("test",     {})
    xgb_test    = xgb_eval.get("test",   {})
    n_total  = int(lda_test.get("total_samples",    0))
    n_sol    = int(lda_test.get("total_solvable",   0))
    n_nosol  = int(lda_test.get("total_unsolvable", 0))

    def _get(d, key, default=0.0):
        v = d.get(key, default)
        return v if v is not None else default
    lda_tn = int(_get(lda_test,  "TN", _get(lda_test,  "CBUG", 0)))
    lda_fp = int(_get(lda_test,  "FP", _get(lda_test,  "IPUG", 0)))
    lda_fn = int(_get(lda_test,  "FN", _get(lda_test,  "WBSG", 0)))
    lda_tp = int(_get(lda_test,  "TP", _get(lda_test,  "CPSG", 0)))
    nn_tn  = int(_get(nn_test,   "TN", _get(nn_test,   "CBUG", 0)))
    nn_fp  = int(_get(nn_test,   "FP", _get(nn_test,   "IPUG", 0)))
    nn_fn  = int(_get(nn_test,   "FN", _get(nn_test,   "WBSG", 0)))
    nn_tp  = int(_get(nn_test,   "TP", _get(nn_test,   "CPSG", 0)))
    xgb_tn = int(_get(xgb_test,  "TN", _get(xgb_test,  "CBUG", 0)))
    xgb_fp = int(_get(xgb_test,  "FP", _get(xgb_test,  "IPUG", 0)))
    xgb_fn = int(_get(xgb_test,  "FN", _get(xgb_test,  "WBSG", 0)))
    xgb_tp = int(_get(xgb_test,  "TP", _get(xgb_test,  "CPSG", 0)))

    lda_specificity = float(_get(lda_test,  "specificity",         _get(lda_test, "cbug_rate", 0.0)))
    lda_accuracy    = float(_get(lda_test,  "accuracy",            _get(lda_test, "solvable_grids_rate", 0.0)))
    lda_error_rate  = float(_get(lda_test,  "error_rate",          _get(lda_test, "unsolvable_grids_rate", 1.0)))
    lda_recall      = float(_get(lda_test,  "recall",              _get(lda_test, "cpsg_rate", 0.0)))
    lda_fn_rate     = float(_get(lda_test,  "fn_rate",             _get(lda_test, "wbsg_rate", 1.0)))
    lda_kb          = float(_get(lda_eval,  "model_size_kb",        0.0))
    lda_ms          = float(_get(lda_test,  "inference_ms_per_sample", 0.0))

    nn_specificity  = float(_get(nn_test,   "specificity",         _get(nn_test,  "cbug_rate", 0.0)))
    nn_accuracy     = float(_get(nn_test,   "accuracy",            _get(nn_test,  "solvable_grids_rate", 0.0)))
    nn_error_rate   = float(_get(nn_test,   "error_rate",          _get(nn_test,  "unsolvable_grids_rate", 1.0)))
    nn_recall       = float(_get(nn_test,   "recall",              _get(nn_test,  "cpsg_rate", 0.0)))
    nn_fn_rate      = float(_get(nn_test,   "fn_rate",             _get(nn_test,  "wbsg_rate", 1.0)))
    nn_kb           = float(_get(nn_eval,   "model_size_kb",        0.0))
    nn_ms           = float(_get(nn_test,   "inference_ms_per_sample", 0.0))
    nn_pw           = nn_eval.get("pos_weight",   "n/a")
    nn_arch         = nn_eval.get("architecture", "n/a")
    nn_input_dim    = nn_eval.get("input_dim",    "n/a")

    xgb_specificity = float(_get(xgb_test,  "specificity",         _get(xgb_test, "cbug_rate", 0.0)))
    xgb_accuracy    = float(_get(xgb_test,  "accuracy",            _get(xgb_test, "solvable_grids_rate", 0.0)))
    xgb_error_rate  = float(_get(xgb_test,  "error_rate",          _get(xgb_test, "unsolvable_grids_rate", 1.0)))
    xgb_recall      = float(_get(xgb_test,  "recall",              _get(xgb_test, "cpsg_rate", 0.0)))
    xgb_fn_rate     = float(_get(xgb_test,  "fn_rate",             _get(xgb_test, "wbsg_rate", 1.0)))
    xgb_kb          = float(_get(xgb_eval,  "model_size_kb",        0.0))
    xgb_ms          = float(_get(xgb_test,  "inference_ms_per_sample", 0.0))

    lda_ok  = (lda_specificity >= recall_min) and (lda_fn_rate  <= fpr_max)
    nn_ok   = (nn_specificity  >= recall_min) and (nn_fn_rate   <= fpr_max)
    xgb_ok  = (xgb_specificity >= recall_min) and (xgb_fn_rate  <= fpr_max)

    passing = [m for m, ok in [("lda", lda_ok), ("xgboost", xgb_ok),
                                ("mlp", nn_ok)] if ok]

    if len(passing) >= 1:
        recommendation = passing[0]
        if len(passing) == 3:
            reason = ("All three models meet the success criteria. "
                      "LDA is preferred — simplest, fastest, most interpretable.")
        elif len(passing) == 2:
            reason = (f"{passing[0].upper()} and {passing[1].upper()} meet the criteria. "
                      f"{passing[0].upper()} preferred for simplicity.")
        else:
            labels = {"lda": "LDA", "xgboost": "XGBoost", "mlp": "MLP"}
            reason = (f"Only {labels[passing[0]]} meets both criteria.")
    else:
        recommendation = "neither"
        sf_lda  = _shortfall_str("LDA",     lda_specificity, lda_fn_rate,  recall_min, fpr_max)
        sf_nn   = _shortfall_str("MLP",     nn_specificity,  nn_fn_rate,   recall_min, fpr_max)
        sf_xgb  = _shortfall_str("XGBoost", xgb_specificity, xgb_fn_rate,  recall_min, fpr_max)
        reason  = (f"No model meets the success criteria. "
                   f"{sf_lda}. {sf_xgb}. {sf_nn}.")
    W = 82

    def _row_int(label, v1, v2, v3, note=""):
        lw = 38; cw = 12
        ns = f"  ← {note}" if note else ""
        return f"  {label:<{lw}}{str(v1):<{cw}}{str(v2):<{cw}}{str(v3):<{cw}}{ns}"

    def _row_pct(label, v1, v2, v3, note=""):
        lw = 38; cw = 12
        ns = f"  ← {note}" if note else ""
        return (f"  {label:<{lw}}{_fmt_pct(v1):<{cw}}"
                f"{_fmt_pct(v2):<{cw}}{_fmt_pct(v3):<{cw}}{ns}")

    print()
    print("═" * W)
    print(f"  RAGATE MODEL COMPARISON — Test Set")
    print(f"  N={n_total:,}  |  Solvable={n_sol:,}  |  Unsolvable={n_nosol:,}")
    print("═" * W)
    from classifier_utils import gate_metrics

    def _added(d, tn, fp, fn, tp):
        m = gate_metrics(TN=tn, FP=fp, FN=fn, TP=tp)
        return {k: d.get(k, m[k]) for k in ("precision", "f1", "balanced_accuracy", "mcc")}

    lda_a = _added(lda_test, lda_tn, lda_fp, lda_fn, lda_tp)
    xgb_a = _added(xgb_test, xgb_tn, xgb_fp, xgb_fn, xgb_tp)
    nn_a  = _added(nn_test,  nn_tn,  nn_fp,  nn_fn,  nn_tp)

    def _mcc(v):
        return "n/a" if v is None else f"{v:.4f}"

    print(f"  {'Outcome Counts':<38}{'LDA':<12}{'XGBoost':<12}{'MLP':<12}")
    print("─" * W)
    print(_row_int("TP  (solvable,   admitted)", lda_tp, xgb_tp, nn_tp))
    print(_row_int("TN  (unsolvable, blocked)",  lda_tn, xgb_tn, nn_tn))
    print(_row_int("FP  (unsolvable, admitted)", lda_fp, xgb_fp, nn_fp))
    print(_row_int("FN  (solvable,   blocked)",  lda_fn, xgb_fn, nn_fn))
    print("─" * W)
    print(f"  {'Metrics':<38}{'LDA':<12}{'XGBoost':<12}{'MLP':<12}")
    print("─" * W)
    print(_row_pct("Recall (sensitivity) TP/(TP+FN)",   lda_recall,      xgb_recall,      nn_recall))
    print(_row_pct("Specificity          TN/(TN+FP)",   lda_specificity, xgb_specificity, nn_specificity, "primary"))
    print(_row_pct("FN Rate (miss rate)  FN/(TP+FN)",   lda_fn_rate,     xgb_fn_rate,     nn_fn_rate,     "primary gate"))
    print(_row_pct("Accuracy RGSR        (TP+TN)/N",    lda_accuracy,    xgb_accuracy,    nn_accuracy,    "overall correct"))
    print(_row_pct("Error Rate RGFR      (FP+FN)/N",    lda_error_rate,  xgb_error_rate,  nn_error_rate))
    print(_row_pct("Precision            TP/(TP+FP)",   lda_a["precision"], xgb_a["precision"], nn_a["precision"]))
    print(_row_pct("F1                   2TP/(2TP+FP+FN)", lda_a["f1"], xgb_a["f1"], nn_a["f1"]))
    print(_row_pct("Balanced accuracy    (Rec+Spec)/2", lda_a["balanced_accuracy"],
                   xgb_a["balanced_accuracy"], nn_a["balanced_accuracy"]))
    print(f"  {'MCC':<38}{_mcc(lda_a['mcc']):<12}{_mcc(xgb_a['mcc']):<12}{_mcc(nn_a['mcc']):<12}")
    print(f"  {'Model Size':<38}{lda_kb:.1f} KB     {xgb_kb:.1f} KB     {nn_kb:.1f} KB")
    print(f"  {'Inference Time (ms/sample)':<38}"
          f"{lda_ms:.4f} ms   {xgb_ms:.4f} ms   {nn_ms:.4f} ms")
    print(f"  {'LDA Architecture':<38}{'Linear Discriminant Analysis'}")
    print(f"  {'XGBoost Architecture':<38}{'Gradient Boosted Trees'}")
    print(f"  {'MLP Architecture':<38}{str(nn_arch)}")
    _parts = [n for n, k in (("mean", "ragate_use_mean_emb"), ("start", "ragate_use_start_emb"),
                             ("goal", "ragate_use_goal_emb")) if config.get(k, n == "mean")]
    print(f"  {'Embedding Dim':<38}{str(nn_input_dim)}  ({'+'.join(_parts)})")
    print(f"  {'pos_weight (MLP)':<38}{'n/a':<12}{'n/a':<12}{str(nn_pw):<12}")
    print("─" * W)
    print(f"  Success criteria: Specificity >= {recall_min*100:.0f}%  "
          f"AND  FN Rate <= {fpr_max*100:.0f}%")
    print(f"  LDA meets criteria:            {'YES' if lda_ok  else 'NO'}")
    print(f"  XGBoost meets criteria:        {'YES' if xgb_ok  else 'NO'}")
    print(f"  MLP meets criteria:            {'YES' if nn_ok   else 'NO'}")
    print("─" * W)
    print(f"  RECOMMENDATION:  {recommendation.upper()}")
    print(f"  REASON:          {reason}")
    print("═" * W)
    print()

    comparison = {
        "model_lda":            lda_test,
        "model_xgboost":        xgb_test,
        "model_mlp": nn_test,
        "recommendation":       recommendation,
        "reason":               reason,
        "success_criteria": {
            "specificity_min": recall_min,
            "fn_rate_max":     fpr_max,
        },
    }

    out_path = config["ragate_comparison_output"]
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"--> Comparison saved to {out_path}")

if __name__ == "__main__":
    main()
