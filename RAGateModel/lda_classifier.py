import os
import sys
import json
import time
import pickle
import argparse

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from classifier_utils import (
    _load_embeddings, _apply_thresholds,
    _compute_metrics, _threshold_sweep, append_gate_log,
)


# ── Random seed ───────────────────────────────────────────────────────────────
_SEED = config.get("random_seed", 42)
np.random.seed(_SEED)

# =============================================================================
# SECTION C — INFERENCE FUNCTION  (importable at runtime)
# =============================================================================

def predict_RAGate_lda(embedding, model, config):
    """
    Run the LDA RAGate check on a single context mean embedding.

    Args:
        embedding (numpy.ndarray): shape [dim] — the context_mean from
            direction_model.get_context_embeddings(), squeezed to [dim]
            and converted to numpy via .cpu().numpy().
        model:    the loaded LDA model (loaded from .pkl at startup)
        config:   the project config dict

    Returns:
        outcome (str):      "Route Available" | "No Route Available"
        probability (float): P(sol | embedding) in [0, 1]

    Threshold logic:
        P(sol|emb) >= tau  →  "Route Available"    (sent to rollout)
        P(sol|emb) <  tau  →  "No Route Available" (blocked)
        tau1_active = False →  always "Route Available"
    """
    tau        = float(config["ragate_model_tau1"])
    tau1_active = bool(config.get("ragate_model_tau1_active", True))

    prob = float(model.predict_proba(embedding.reshape(1, -1))[0, 1])

    if tau1_active and prob < tau:
        return "No Route Available", prob
    return "Route Available", prob

# =============================================================================
# SECTION A — TRAINING
# =============================================================================

def train_lda():
    """Fit LDA on training embeddings and save the model."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    print("LDA --> ── Training ──────────────────────────────────────────────")

    train_path = config["ragate_embeddings_train"]
    X_train, y_train, _ = _load_embeddings(train_path)
    print(f"LDA --> Training embeddings loaded: {X_train.shape}")

    if "ragate_lda_priors" not in config:
        raise KeyError(
            "LDA --> config key 'ragate_lda_priors' is missing. "
            "Add it to config.py, e.g.: [0.07, 0.93]"
        )
    priors = config["ragate_lda_priors"]

    t0  = time.perf_counter()
    lda = LinearDiscriminantAnalysis(solver="svd", priors=priors)
    lda.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    model_path = config["ragate_lda_model_path"]
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
                exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(lda, f)

    model_kb = os.path.getsize(model_path) / 1024.0
    n_sol    = int((y_train == 1).sum())
    n_nosol  = int((y_train == 0).sum())

    print(f"LDA --> Training complete  ({elapsed:.1f}s)")
    print(f"LDA --> Training samples: {len(y_train):,}  "
          f"(solvable={n_sol:,}  unsolvable={n_nosol:,})")
    print(f"LDA --> Priors: {priors}")
    print(f"LDA --> Model saved to {model_path}")
    print(f"LDA --> Model size: {model_kb:.1f} KB")


# =============================================================================
# SECTION B — EVALUATION
# =============================================================================

def evaluate_lda():
    """Evaluate the saved LDA model on val and test splits."""

    print("LDA --> ── Evaluation ────────────────────────────────────────────")

    model_path = config["ragate_lda_model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LDA --> Model not found: {model_path}\n"
            f"      Run: python lda_classifier.py --train"
        )
    with open(model_path, "rb") as f:
        lda = pickle.load(f)
    model_kb    = os.path.getsize(model_path) / 1024.0
    print(f"LDA --> Model loaded from {model_path}  ({model_kb:.1f} KB)")

    tau        = float(config["ragate_model_tau1"])
    tau1_active = bool(config.get("ragate_model_tau1_active", True))

    eval_results = {
        "model":         "lda",
        "tau1_active":   tau1_active,
        "tau":          tau,
        "model_size_kb": round(model_kb, 2),
    }
    sweep_results = {"model": "lda"}

    splits = [
        ("val",  config["ragate_embeddings_val"]),
        ("test", config["ragate_embeddings_test"]),
    ]

    for split_name, emb_path in splits:
        X, y, _ = _load_embeddings(emb_path)

        t_inf_start   = time.perf_counter()
        probs         = lda.predict_proba(X)[:, 1]
        t_inf_end     = time.perf_counter()
        ms_per_sample = (t_inf_end - t_inf_start) * 1000.0 / max(len(X), 1)

        _, pred_nosol, pred_sol = _apply_thresholds(probs, tau, tau1_active)
        metrics = _compute_metrics(y, probs, pred_nosol, pred_sol)
        metrics["inference_ms_per_sample"] = round(ms_per_sample, 4)
        eval_results[split_name] = metrics
        append_gate_log("lda", split_name, metrics, tau, tau1_active,
                        model_size_kb=round(model_kb, 2), embedding_dim=X.shape[1])

        sweep_rows = _threshold_sweep(y, probs, tau1_active)
        sweep_results[f"{split_name}_sweep"] = sweep_rows

        # ── Printed report ────────────────────────────────────────────────────
        print(f"\nLDA Classifier === Evaluation: {split_name} ===")
        print(f"  Outcome Counts:")
        print(f"    Correctly Blocked Unsolvable Grid  (CBUG): {metrics['CBUG']:,}")
        print(f"    Incorrectly Processed Unsolvable Grid (IPUG): {metrics['IPUG']:,}")
        print(f"    Wrongly Blocked Solvable Grid      (WBSG): {metrics['WBSG']:,}")
        print(f"    Correctly Processed Solvable Grid  (CPSG): {metrics['CPSG']:,}")
        print(f"  Metrics:")
        print(f"    Correctly Blocked Unsolvable Grid Rate  (CBUG Rate): "
              f"{metrics['cbug_rate']:.4f}")
        print(f"    Correctly Processed Solvable Grid Rate  (CPSG Rate): "
              f"{metrics['cpsg_rate']:.4f}")
        print(f"    Wrongly Blocked Solvable Grid Rate      (WBSG Rate): "
              f"{metrics['wbsg_rate']:.4f}  ← primary gate")
        print(f"    Solvable Grids Rate                  (SGR):       "
              f"{metrics['solvable_grids_rate']:.4f}  ← overall correct decision rate")
        print(f"    Unsolvable Grids Rate                (UGR):       "
              f"{metrics['unsolvable_grids_rate']:.4f}")
        print(f"    Inference: {ms_per_sample:.4f} ms/sample")

    # ── Save evaluation JSON ──────────────────────────────────────────────────
    eval_path = config["ragate_lda_eval_output"]
    os.makedirs(os.path.dirname(eval_path) if os.path.dirname(eval_path) else ".",
                exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n--> LDA Classifier - Results saved to {eval_path}")

    # ── Save sweep JSON ───────────────────────────────────────────────────────
    sweep_path = config["ragate_lda_sweep_output"]
    os.makedirs(os.path.dirname(sweep_path) if os.path.dirname(sweep_path) else ".",
                exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"--> LDA Classifier - Sweep saved to {sweep_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LDA RAGate classifier — train or evaluate"
    )
    parser.add_argument("--train",    action="store_true", help="Fit and save LDA model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved LDA model")
    args = parser.parse_args()

    if not args.train and not args.evaluate:
        parser.print_help()
        sys.exit(0)

    if args.train:
        train_lda()

    if args.evaluate:
        evaluate_lda()
