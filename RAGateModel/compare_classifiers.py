
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config

# =============================================================================
# HELPERS
# =============================================================================

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
    if isinstance(value, float):
        return f"{value * 100:.2f}%"
    return str(value)


def _shortfall_str(model_label, cbug_rate, wbsg_rate, recall_min, fpr_max):
    parts = []
    if cbug_rate < recall_min:
        parts.append(f"CBUG Rate={cbug_rate*100:.2f}% < {recall_min*100:.0f}%")
    if wbsg_rate > fpr_max:
        parts.append(f"WBSG Rate={wbsg_rate*100:.2f}% > {fpr_max*100:.0f}%")
    return f"{model_label}: {', '.join(parts)}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    recall_min = float(config.get("ragate_success_recall_min", 0.95))
    fpr_max    = float(config.get("ragate_success_fpr_max",    0.02))

    # ── Load eval JSONs ───────────────────────────────────────────────────────
    lda_eval    = _load_eval_json(config["ragate_lda_eval_output"],      "LDA")
    nn_eval     = _load_eval_json(config["ragate_mlp_eval_output"],   "MLP")
    xgb_eval    = _load_eval_json(config["ragate_xgboost_eval_output"],  "XGBoost")

    lda_test    = lda_eval.get("test",    {})
    nn_test     = nn_eval.get("test",     {})
    xgb_test    = xgb_eval.get("test",   {})

    # Actual test-set size from loaded data
    n_total  = int(lda_test.get("total_samples",    0))
    n_sol    = int(lda_test.get("total_solvable",   0))
    n_nosol  = int(lda_test.get("total_unsolvable", 0))

    def _get(d, key, default=0.0):
        v = d.get(key, default)
        return v if v is not None else default

    # ── Outcome counts ────────────────────────────────────────────────────────
    lda_cbug = int(_get(lda_test,  "CBUG", 0))
    lda_ipug = int(_get(lda_test,  "IPUG", 0))
    lda_wbsg = int(_get(lda_test,  "WBSG", 0))
    lda_cpsg = int(_get(lda_test,  "CPSG", 0))
    nn_cbug  = int(_get(nn_test,   "CBUG", 0))
    nn_ipug  = int(_get(nn_test,   "IPUG", 0))
    nn_wbsg  = int(_get(nn_test,   "WBSG", 0))
    nn_cpsg  = int(_get(nn_test,   "CPSG", 0))
    xgb_cbug = int(_get(xgb_test,  "CBUG", 0))
    xgb_ipug = int(_get(xgb_test,  "IPUG", 0))
    xgb_wbsg = int(_get(xgb_test,  "WBSG", 0))
    xgb_cpsg = int(_get(xgb_test,  "CPSG", 0))

    # ── Metrics ───────────────────────────────────────────────────────────────
    lda_cbug_rate  = float(_get(lda_test,  "cbug_rate",           0.0))
    lda_sgr        = float(_get(lda_test,  "solvable_grids_rate",   0.0))
    lda_ugr        = float(_get(lda_test,  "unsolvable_grids_rate", 1.0))
    lda_cpsg_rate  = float(_get(lda_test,  "cpsg_rate",           0.0))
    lda_wbsg_rate  = float(_get(lda_test,  "wbsg_rate",           1.0))
    lda_kb         = float(_get(lda_eval,  "model_size_kb",        0.0))
    lda_ms         = float(_get(lda_test,  "inference_ms_per_sample", 0.0))

    nn_cbug_rate   = float(_get(nn_test,   "cbug_rate",           0.0))
    nn_sgr         = float(_get(nn_test,   "solvable_grids_rate",   0.0))
    nn_ugr         = float(_get(nn_test,   "unsolvable_grids_rate", 1.0))
    nn_cpsg_rate   = float(_get(nn_test,   "cpsg_rate",           0.0))
    nn_wbsg_rate   = float(_get(nn_test,   "wbsg_rate",           1.0))
    nn_kb          = float(_get(nn_eval,   "model_size_kb",        0.0))
    nn_ms          = float(_get(nn_test,   "inference_ms_per_sample", 0.0))
    nn_pw          = nn_eval.get("pos_weight",   "n/a")
    nn_arch        = nn_eval.get("architecture", "n/a")
    nn_input_dim   = nn_eval.get("input_dim",    "n/a")

    xgb_cbug_rate  = float(_get(xgb_test,  "cbug_rate",           0.0))
    xgb_sgr        = float(_get(xgb_test,  "solvable_grids_rate",   0.0))
    xgb_ugr        = float(_get(xgb_test,  "unsolvable_grids_rate", 1.0))
    xgb_cpsg_rate  = float(_get(xgb_test,  "cpsg_rate",           0.0))
    xgb_wbsg_rate  = float(_get(xgb_test,  "wbsg_rate",           1.0))
    xgb_kb         = float(_get(xgb_eval,  "model_size_kb",        0.0))
    xgb_ms         = float(_get(xgb_test,  "inference_ms_per_sample", 0.0))

    # ── Success criteria ──────────────────────────────────────────────────────
    lda_ok  = (lda_cbug_rate  >= recall_min) and (lda_wbsg_rate  <= fpr_max)
    nn_ok   = (nn_cbug_rate   >= recall_min) and (nn_wbsg_rate   <= fpr_max)
    xgb_ok  = (xgb_cbug_rate  >= recall_min) and (xgb_wbsg_rate  <= fpr_max)

    # ── Recommendation ────────────────────────────────────────────────────────
    passing = [m for m, ok in [("lda", lda_ok), ("xgboost", xgb_ok),
                                ("mlp", nn_ok)] if ok]

    if len(passing) >= 1:
        # Priority: lda > xgboost > mlp (simplest/fastest first)
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
        sf_lda  = _shortfall_str("LDA",            lda_cbug_rate,  lda_wbsg_rate,  recall_min, fpr_max)
        sf_nn   = _shortfall_str("MLP", nn_cbug_rate,   nn_wbsg_rate,   recall_min, fpr_max)
        sf_xgb  = _shortfall_str("XGBoost",        xgb_cbug_rate,  xgb_wbsg_rate,  recall_min, fpr_max)
        reason  = (f"No model meets the success criteria. "
                   f"{sf_lda}. {sf_xgb}. {sf_nn}.")

    # ── Print comparison table ────────────────────────────────────────────────
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
    print(f"  {'Outcome Counts':<38}{'LDA':<12}{'XGBoost':<12}{'MLP':<12}")
    print("─" * W)
    print(_row_int("Correctly Blocked Unsolvable Grid     (CBUG)",
                   lda_cbug, xgb_cbug, nn_cbug))
    print(_row_int("Incorrectly Processed Unsolvable Grid (IPUG)",
                   lda_ipug, xgb_ipug, nn_ipug))
    print(_row_int("Wrongly Blocked Solvable Grid         (WBSG)",
                   lda_wbsg, xgb_wbsg, nn_wbsg))
    print(_row_int("Correctly Processed Solvable Grid     (CPSG)",
                   lda_cpsg, xgb_cpsg, nn_cpsg))
    print("─" * W)
    print(f"  {'Metrics':<38}{'LDA':<12}{'XGBoost':<12}{'MLP':<12}")
    print("─" * W)
    print(_row_pct("CBUG Rate",
                   lda_cbug_rate, xgb_cbug_rate, nn_cbug_rate, "primary"))
    print(_row_pct("CPSG Rate",
                   lda_cpsg_rate, xgb_cpsg_rate, nn_cpsg_rate))
    print(_row_pct("WBSG Rate",
                   lda_wbsg_rate, xgb_wbsg_rate, nn_wbsg_rate, "primary gate"))
    print(_row_pct("Solvable Grids Rate (SGR)",
                   lda_sgr, xgb_sgr, nn_sgr, "overall correct"))
    print(_row_pct("Unsolvable Grids Rate (UGR)",
                   lda_ugr, xgb_ugr, nn_ugr))
    print(f"  {'Model Size':<38}{lda_kb:.1f} KB     {xgb_kb:.1f} KB     {nn_kb:.1f} KB")
    print(f"  {'Inference Time (ms/sample)':<38}"
          f"{lda_ms:.4f} ms   {xgb_ms:.4f} ms   {nn_ms:.4f} ms")
    print(f"  {'LDA Architecture':<38}{'Linear Discriminant Analysis'}")
    print(f"  {'XGBoost Architecture':<38}{'Gradient Boosted Trees'}")
    print(f"  {'MLP Architecture':<38}{str(nn_arch)}")
    print(f"  {'MLP Embedding Dim':<38}{str(nn_input_dim)}"
          f"  ({'mean only' if nn_input_dim == 512 else 'mean+start+goal' if nn_input_dim == 1536 else ''})")
    print(f"  {'pos_weight (MLP)':<38}{'n/a':<12}{'n/a':<12}{str(nn_pw):<12}")
    print("─" * W)
    print(f"  Success criteria: CBUG Rate >= {recall_min*100:.0f}%  "
          f"AND  WBSG Rate <= {fpr_max*100:.0f}%")
    print(f"  LDA meets criteria:            {'YES' if lda_ok  else 'NO'}")
    print(f"  XGBoost meets criteria:        {'YES' if xgb_ok  else 'NO'}")
    print(f"  MLP meets criteria: {'YES' if nn_ok   else 'NO'}")
    print("─" * W)
    print(f"  RECOMMENDATION:  {recommendation.upper()}")
    print(f"  REASON:          {reason}")
    print("═" * W)
    print()

    # ── Save comparison JSON ──────────────────────────────────────────────────
    comparison = {
        "model_lda":            lda_test,
        "model_xgboost":        xgb_test,
        "model_mlp": nn_test,
        "recommendation":       recommendation,
        "reason":               reason,
        "success_criteria": {
            "cbug_rate_min": recall_min,
            "wbsg_rate_max": fpr_max,
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
