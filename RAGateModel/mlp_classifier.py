import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from classifier_utils import (
    _load_embeddings, _apply_thresholds,
    _compute_metrics, _threshold_sweep, append_gate_log,
)


# ── Random seed ───────────────────────────────────────────────────────────────
_SEED = config.get("random_seed", 42)
torch.manual_seed(_SEED)
np.random.seed(_SEED)


# =============================================================================
# NETWORK DEFINITION
# =============================================================================

class RAGateMLPNet(nn.Module):
    def __init__(self, input_dim: int, dropout: float, hidden_layers: list):
        super().__init__()

        if not hidden_layers:
            raise ValueError(
                "ragate_mlp_hidden_layers must have at least one size."
            )

        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final output layer — raw logit, no activation
        layers.append(nn.Linear(in_dim, 1))

        self.net        = nn.Sequential(*layers)
        self.arch_str   = (f"{input_dim}→"
                           + "→".join(str(h) for h in hidden_layers)
                           + "→1")

    def forward(self, x):
        return self.net(x)


# =============================================================================
# SECTION C — INFERENCE FUNCTION  (importable at runtime)
# =============================================================================

def predict_ragate_mlp(embedding, model, config, device):

    tau        = float(config["ragate_model_tau1"])
    tau1_active = bool(config.get("ragate_model_tau1_active", True))

    with torch.no_grad():
        inp    = embedding.unsqueeze(0).to(device)
        logit  = model(inp).squeeze()
        prob   = float(torch.sigmoid(logit).item())   # Sigmoid applied here

    if tau1_active and prob < tau:
        return "No Route Available", prob
    return "Route Available", prob


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _get_probs_from_net(net, X_tensor, device, batch_size=4096):
    net.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb    = X_tensor[i:i + batch_size].to(device)
            logits = net(xb).squeeze(1)                    # raw logits
            preds  = torch.sigmoid(logits).cpu().numpy()   # → probabilities
            all_probs.append(preds)
    return np.concatenate(all_probs)


def _eval_on_split(net, X_tensor, y_tensor, device, tau, tau1_active):

    y_np = y_tensor.numpy()

    t_start = time.perf_counter()
    probs   = _get_probs_from_net(net, X_tensor, device)
    t_end   = time.perf_counter()

    ms_per_sample = (t_end - t_start) * 1000.0 / max(len(X_tensor), 1)

    _, pred_nosol, pred_sol = _apply_thresholds(probs, tau, tau1_active)
    metrics = _compute_metrics(y_np, probs, pred_nosol, pred_sol)
    metrics["inference_ms_per_sample"] = round(ms_per_sample, 4)
    return metrics, probs


# =============================================================================
# SECTION A — TRAINING
# =============================================================================

def train_mlp():
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"MLP --> Device: {device}")
    print("MLP --> ── Training ───────────────────────────────────────────")

    lr            = float(config.get("ragate_mlp_lr",           1e-4))
    n_epochs      = int(config.get("ragate_mlp_epochs",         50))
    dropout       = float(config.get("ragate_mlp_dropout",      0.2))
    weight_decay  = float(config.get("ragate_mlp_weight_decay", 1e-4))
    batch_size    = int(config.get("ragate_mlp_batch_size",     128))
    pos_weight    = float(config.get("ragate_mlp_pos_weight",   0.5))
    if "ragate_mlp_hidden_layers" not in config:
        raise KeyError(
            "MLP --> config key 'ragate_mlp_hidden_layers' is missing. "
            "Add it to config.py, e.g.: [256, 64] or [512, 256, 64]"
        )
    hidden_layers = list(config["ragate_mlp_hidden_layers"])
    tau          = float(config["ragate_model_tau1"])
    tau1_active   = bool(config.get("ragate_model_tau1_active", True))

    X_train, y_train, _ = _load_embeddings(config["ragate_embeddings_train"])
    X_val,   y_val,   _ = _load_embeddings(config["ragate_embeddings_val"])

    # input_dim from actual data — handles both 512 (mean) and 1536 (mean+start+goal)
    input_dim = X_train.shape[1]
    print(f"MLP --> Train: {len(X_train):,} samples  |  Val: {len(X_val):,} samples")
    print(f"MLP --> Embedding dim: {input_dim} "
          f"({'mean+start+goal' if input_dim > 512 else 'mean only'})")

    # Convert numpy arrays to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val   = torch.from_numpy(X_val).float()
    y_val   = torch.from_numpy(y_val).float()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader  = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    net = RAGateMLPNet(
        input_dim=input_dim,
        dropout=dropout,
        hidden_layers=hidden_layers,
    ).to(device)
    print(f"MLP --> Architecture: {net.arch_str}")

    pw_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    print(f"MLP --> Loss: BCEWithLogitsLoss  pos_weight={pos_weight}"
          f"  ({'conservative — blocks more' if pos_weight < 1.0 else 'permissive — passes more' if pos_weight > 1.0 else 'standard BCE'})")
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )

    model_path = config["ragate_mlp_model_path"]
    os.makedirs(
        os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
        exist_ok=True
    )

    best_val_f1  = -1.0
    best_val_rec = 0.0
    best_epoch   = 0
    training_rows = []

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        net.train()
        running_loss = 0.0
        n_batches    = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = net(X_batch).squeeze(1)    
            loss   = criterion(logits, y_batch) 
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches    += 1
        train_loss = running_loss / max(n_batches, 1)

        # ── Val evaluation ────────────────────────────────────────────────────
        val_metrics, _ = _eval_on_split(
            net, X_val, y_val, device, tau, tau1_active
        )
        val_sgr       = val_metrics["solvable_grids_rate"]
        val_cbug_rate = val_metrics["cbug_rate"]
        val_wbsg_rate = val_metrics["wbsg_rate"]

        # ── Save best checkpoint by val Solvable Grids Rate ────────────────
        if val_sgr > best_val_f1:
            best_val_f1  = val_sgr
            best_val_rec = val_cbug_rate
            best_epoch   = epoch
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_sgr":              val_sgr,
                "val_cbug_rate":        val_cbug_rate,
                "pos_weight":           pos_weight,
                "hidden_layers":        hidden_layers,
                "arch_str":             net.arch_str,
                "input_dim":            input_dim,
            }, model_path)

        # ── Log row ───────────────────────────────────────────────────────────
        training_rows.append({
            "epoch":          epoch,
            "train_loss":     round(train_loss,    6),
            "pos_weight":     pos_weight,
            "val_sgr":        round(val_sgr,        4),
            "val_cbug_rate":  round(val_cbug_rate, 4),
            "val_wbsg_rate":  round(val_wbsg_rate, 4),
        })

        # ── Per-epoch print ───────────────────────────────────────────────────
        print(f"--> MLP Ep{epoch}/{n_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"pos_weight: {pos_weight} | "
              f"Val SGR: {val_sgr:.4f} | "
              f"Val CBUG Rate: {val_cbug_rate:.4f} | "
              f"Val WBSG Rate: {val_wbsg_rate:.4f}")

    # ── Save training log CSV ─────────────────────────────────────────────────
    log_path = config["ragate_mlp_training_log"]
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".",
                exist_ok=True)
    fieldnames = ["epoch", "train_loss", "pos_weight",
                  "val_sgr", "val_cbug_rate", "val_wbsg_rate"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_rows)

    print(f"\n--> MLP Classifier - Training complete")
    print(f"--> MLP Classifier - Architecture: {net.arch_str}")
    print(f"--> MLP Classifier - Best Val SGR: {best_val_f1:.4f} at Ep{best_epoch}")
    print(f"--> MLP Classifier - Checkpoint saved to {model_path}")


# =============================================================================
# SECTION B — EVALUATION
# =============================================================================

def evaluate_mlp():
    """Evaluate the saved binary model checkpoint on val and test splits."""

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"MLP --> Device: {device}")
    print("MLP --> ── Evaluation ─────────────────────────────────────────")

    tau        = float(config["ragate_model_tau1"])
    tau1_active = bool(config.get("ragate_model_tau1_active", True))
    dropout     = float(config.get("ragate_mlp_dropout", 0.2))

    model_path = config["ragate_mlp_model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MLP --> Checkpoint not found: {model_path}\n"
            f"         Run: python mlp_classifier.py --train"
        )
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    # input_dim from checkpoint (saved at training time) — handles 512 and 1536
    input_dim   = int(checkpoint.get("input_dim", config["dim"]))
    hidden_layers = checkpoint.get(
        "hidden_layers",
        list(config["ragate_mlp_hidden_layers"])
    )
    net = RAGateMLPNet(
        input_dim=input_dim,
        dropout=dropout,
        hidden_layers=hidden_layers,
    ).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    model_kb = os.path.getsize(model_path) / 1024.0
    print(f"MLP --> Model loaded from {model_path}  ({model_kb:.1f} KB)")
    print(f"MLP --> Architecture: {checkpoint.get('arch_str', 'unknown')}")
    print(f"MLP --> Best epoch: {checkpoint.get('epoch', '?')}  "
          f"Val SGR={checkpoint.get('val_sgr', checkpoint.get('val_f1', '?'))}  "
          f"pos_weight={checkpoint.get('pos_weight', 'n/a')}")

    eval_results = {
        "model":         "mlp",
        "architecture":  checkpoint.get("arch_str", "unknown"),
        "input_dim":     input_dim,
        "tau1_active":   tau1_active,
        "tau":          tau,
        "model_size_kb": round(model_kb, 2),
        "pos_weight":    checkpoint.get("pos_weight", "n/a"),
    }
    sweep_results = {"model": "mlp"}

    splits = [
        ("val",  config["ragate_embeddings_val"]),
        ("test", config["ragate_embeddings_test"]),
    ]

    for split_name, emb_path in splits:
        X_np, y_np, _ = _load_embeddings(emb_path)
        X = torch.from_numpy(X_np).float()
        y = torch.from_numpy(y_np).float()

        metrics, probs = _eval_on_split(
            net, X, y, device, tau, tau1_active
        )
        eval_results[split_name] = metrics
        append_gate_log("mlp", split_name, metrics, tau, tau1_active,
                        model_size_kb=round(model_kb, 2), embedding_dim=input_dim)

        sweep_rows = _threshold_sweep(y_np, probs)
        sweep_results[f"{split_name}_sweep"] = sweep_rows

        # ── Printed report ────────────────────────────────────────────────────
        print(f"\nMLP Classifier === Evaluation: {split_name} ===")
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
        print(f"    Inference: {metrics['inference_ms_per_sample']:.4f} ms/sample")

    # ── Save evaluation JSON ──────────────────────────────────────────────────
    eval_path = config["ragate_mlp_eval_output"]
    os.makedirs(os.path.dirname(eval_path) if os.path.dirname(eval_path) else ".",
                exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n--> MLP Classifier - Results saved to {eval_path}")

    # ── Save sweep JSON ───────────────────────────────────────────────────────
    sweep_path = config["ragate_mlp_sweep_output"]
    os.makedirs(os.path.dirname(sweep_path) if os.path.dirname(sweep_path) else ".",
                exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"--> MLP Classifier - Sweep saved to {sweep_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binary RAGate classifier — train or evaluate"
    )
    parser.add_argument("--train",    action="store_true",
                        help="Train and save best checkpoint")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate saved checkpoint on val + test")
    args = parser.parse_args()

    if not args.train and not args.evaluate:
        parser.print_help()
        sys.exit(0)

    if args.train:
        train_mlp()

    if args.evaluate:
        evaluate_mlp()
