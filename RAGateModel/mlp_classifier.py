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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from classifier_utils import (print_gate_report, 
    _load_embeddings, _apply_thresholds,
    _compute_metrics, _threshold_sweep, append_gate_log,
)

_SEED = config.get("random_seed", 42)
torch.manual_seed(_SEED)
np.random.seed(_SEED)

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

        layers.append(nn.Linear(in_dim, 1))

        self.net        = nn.Sequential(*layers)
        self.arch_str   = (f"{input_dim}→"
                           + "→".join(str(h) for h in hidden_layers)
                           + "→1")

    def forward(self, x):
        return self.net(x)

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

def finetune_transformer():
 
    from utils import set_seed
    set_seed()   
    import torch.utils.data as tud

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"\nJointFT --> Device: {device}")
    print("JointFT --> ── Step 0: Joint Fine-Tuning PSTAR + MLP ─────────────────")

    n_epochs      = int(config.get("ragate_joint_finetune_epochs", 10))
    lr            = float(config.get("ragate_joint_finetune_lr",   1e-5))
    pos_weight    = float(config.get("ragate_mlp_pos_weight",      0.5))
    hidden_layers = list(config.get("ragate_mlp_hidden_layers",    [512, 256, 64]))
    batch_size    = int(config.get("ragate_mlp_batch_size",        128))
    out_ckpt      = config["ragate_joint_finetune_checkpoint"]

    print(f"JointFT --> Epochs: {n_epochs}  LR: {lr}  pos_weight: {pos_weight}")
    print(f"JointFT --> MLP arch: {hidden_layers}")
    print(f"JointFT --> New checkpoint will be saved to: {out_ckpt}")

    train_pt_path = config["ragate_train_pt"]
    if not os.path.exists(train_pt_path):
        raise FileNotFoundError(
            f"JointFT --> Raw training data not found: {train_pt_path}\n"
            f"  Make sure ragate_train_pt points to the correct .pt file."
        )
    raw = torch.load(train_pt_path, map_location="cpu", weights_only=False)
    inputs   = torch.stack([d["input_tensor"]   for d in raw]).float()   # [N, seq, feats]
    masks    = torch.stack([d["masking_tensor"]  for d in raw]).float()   # [N, seq]
    labels   = torch.tensor([int(d["label"])     for d in raw]).float()   # [N]
    n_total  = len(labels)
    print(f"JointFT --> Loaded {n_total:,} raw training samples from {train_pt_path}")

    train_ds = tud.TensorDataset(inputs, masks, labels)
    train_loader = tud.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    from inference import load_model_from_checkpoint
    checkpoint_path = os.path.join(
        config["checkpoint_dir"],
        config["inference_model_name"],
    )
    pstar = load_model_from_checkpoint(checkpoint_path, device)
    pstar.train()
    for p in pstar.parameters():
        p.requires_grad = True
    print(f"JointFT --> PSTAR loaded (unfrozen) from {checkpoint_path}")

    with torch.no_grad():
        probe_result = pstar.get_context_embeddings(
            inputs[:1].to(device), grad=False
        )
    emb_parts  = [v for v in probe_result.values()]
    input_dim  = sum(t.shape[-1] for t in emb_parts)
    pstar.train()
    for p in pstar.parameters():
        p.requires_grad = True
    print(f"JointFT --> Embedding dim (probe): {input_dim}")

    head = RAGateMLPNet(
        input_dim=input_dim,
        dropout=float(config.get("ragate_mlp_dropout", 0.1)),
        hidden_layers=hidden_layers,
    ).to(device)
    print(f"JointFT --> PSTAR to MLP arch: {head.arch_str}")

    optimizer = torch.optim.Adam(
        list(pstar.parameters()) + list(head.parameters()),
        lr=lr,
    )
    pw_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    best_loss        = float("inf")
    best_epoch       = 0
    best_state_dict  = None  

    os.makedirs(
        os.path.dirname(out_ckpt) if os.path.dirname(out_ckpt) else ".",
        exist_ok=True,
    )

    for epoch in range(1, n_epochs + 1):
        pstar.train()
        head.train()
        running_loss = 0.0
        n_batches    = 0

        for x_batch, m_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            emb_dict = pstar.get_context_embeddings(x_batch, grad=True)
            emb      = torch.cat(list(emb_dict.values()), dim=-1)  # [B, input_dim]

            logits = head(emb).squeeze(1)                           # [B]
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches    += 1

        epoch_loss = running_loss / max(n_batches, 1)

        if epoch_loss < best_loss:
            best_loss       = epoch_loss
            best_epoch      = epoch

            import copy
            best_state_dict = copy.deepcopy(pstar.state_dict())
            pstar.eval()
            torch.save(
                {
                    "model_state_dict":      best_state_dict,
                    "head_state_dict":       copy.deepcopy(head.state_dict()),
                    "head_hidden_layers":    hidden_layers,
                    "head_input_dim":        input_dim,
                    "epoch":                 best_epoch,
                    "finetune_epochs":       n_epochs,
                    "finetune_lr":           lr,
                    "finetune_val_loss":     best_loss,
                    "source_checkpoint":     checkpoint_path,
                },
                out_ckpt,
            )
            pstar.train()
            marker = " ← best (saved)"
        else:
            marker = ""

        print(f"--> JointFT Ep{epoch}/{n_epochs} | Loss: {epoch_loss:.6f}{marker}")

    print(f"\nJointFT --> Fine-tuning complete.")
    print(f"JointFT --> Best BCE loss: {best_loss:.6f} at epoch {best_epoch}/{n_epochs}")
    print(f"JointFT --> Fine-tuned PSTAR checkpoint saved → {out_ckpt}")
    print(f"JointFT --> NOTE: Temporary MLP head discarded — it is NOT the RAGate MLP.")
    print(f"JointFT -->       Step 4 will train the permanent RAGate MLP on re-extracted embeddings.")
    print(f"JointFT -->       Ensure ragate_embeddings_force_reextract=True in config.py.")

def evaluate_finetune_head():
    import torch.utils.data as tud

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"\nStep0b --> Device: {device}")
    print("Step0b --> ── Evaluating fine-tuned PSTAR + temp MLP head ───────────────")

    ft_ckpt_path = config["ragate_joint_finetune_checkpoint"]
    if not os.path.exists(ft_ckpt_path):
        raise FileNotFoundError(
            f"Step0b --> Fine-tuned checkpoint not found: {ft_ckpt_path}\n"
            f"           Run Step 0 (ragate_joint_finetune=True) first."
        )

    from inference import load_model_from_checkpoint
    pstar = load_model_from_checkpoint(ft_ckpt_path, device)
    pstar.eval()
    for p in pstar.parameters():
        p.requires_grad = False

    ckpt_meta = torch.load(ft_ckpt_path, map_location="cpu", weights_only=False)
    best_epoch      = ckpt_meta.get("epoch", "?")
    finetune_epochs = ckpt_meta.get("finetune_epochs", "?")
    finetune_lr     = ckpt_meta.get("finetune_lr", "?")
    has_head        = "head_state_dict" in ckpt_meta
    print(f"Step0b --> Loaded fine-tuned PSTAR from {ft_ckpt_path}")
    print(f"Step0b --> Fine-tune: {finetune_epochs} epochs, LR={finetune_lr}, best at ep{best_epoch}")
    print(f"Step0b --> Head weights in checkpoint: {has_head}")

    _feature_size = config.get("feature_size", 4)
    _dummy = torch.zeros(1, 1, _feature_size, device=device)
    with torch.no_grad():
        probe_result = pstar.get_context_embeddings(_dummy, grad=False)
    emb_parts = [v for v in probe_result.values()]
    input_dim = sum(t.shape[-1] for t in emb_parts)
    print(f"Step0b --> Embedding dim: {input_dim}")

    if has_head:
        hidden_layers = list(ckpt_meta.get("head_hidden_layers",
                             config.get("ragate_mlp_hidden_layers", [512, 256, 64])))
        saved_input_dim = int(ckpt_meta.get("head_input_dim", input_dim))
        if saved_input_dim != input_dim:
            print(f"Step0b --> [WARN] head input_dim in checkpoint ({saved_input_dim}) "
                  f"!= probed embedding dim ({input_dim}). Using checkpoint value.")
            input_dim = saved_input_dim
    else:
        hidden_layers = list(config.get("ragate_mlp_hidden_layers", [512, 256, 64]))
        print("Step0b --> [WARN] No head weights found in checkpoint (old format).")
        print("           Head will use random weights — embedding structure only.")

    dropout = float(config.get("ragate_mlp_dropout", 0.1))
    head = RAGateMLPNet(
        input_dim=input_dim,
        dropout=dropout,
        hidden_layers=hidden_layers,
    ).to(device)

    if has_head:
        head.load_state_dict(ckpt_meta["head_state_dict"])
        print(f"Step0b --> Head weights loaded from checkpoint. Arch: {head.arch_str}")
    else:
        print(f"Step0b --> Head randomly initialised. Arch: {head.arch_str}")

    head.eval()

    tau         = float(config["ragate_model_tau1"])
    tau1_active = bool(config.get("ragate_model_tau1_active", True))

    head_note = (
        "Fine-tuned PSTAR + saved Step-0 MLP head (trained weights restored)."
        if has_head else
        "Old checkpoint format: head weights not saved. "
        "Random-weight head used — reflects embedding structure only."
    )
    eval_results = {
        "model":            "pstar_finetune_head",
        "architecture":     head.arch_str,
        "input_dim":        input_dim,
        "tau":              tau,
        "tau1_active":      tau1_active,
        "finetune_epochs":  finetune_epochs,
        "finetune_lr":      finetune_lr,
        "best_epoch":       best_epoch,
        "head_weights_restored": has_head,
        "note":             head_note,
    }

    splits = [
        ("val",  config["ragate_val_pt"]),
        ("test", config["ragate_test_pt"]),
    ]

    for split_name, raw_pt_path in splits:
        if not os.path.exists(raw_pt_path):
            print(f"Step0b --> [SKIP] {split_name} raw .pt not found: {raw_pt_path}")
            continue

        raw = torch.load(raw_pt_path, map_location="cpu", weights_only=False)
        inputs = torch.stack([d["input_tensor"]  for d in raw]).float()
        labels = torch.tensor([int(d["label"])   for d in raw]).float()
        n_total = len(labels)
        print(f"Step0b --> {split_name}: {n_total:,} samples")

        batch_size = int(config.get("ragate_mlp_batch_size", 128))
        all_embs   = []
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                xb = inputs[i:i + batch_size].to(device)
                emb_dict = pstar.get_context_embeddings(xb, grad=False)
                emb = torch.cat(list(emb_dict.values()), dim=-1)   # [B, input_dim]
                all_embs.append(emb.cpu())
        X_tensor = torch.cat(all_embs, dim=0)   # [N, input_dim]
        t_start = time.perf_counter()
        probs   = _get_probs_from_net(head, X_tensor, device)
        t_end   = time.perf_counter()
        ms_per_sample = (t_end - t_start) * 1000.0 / max(n_total, 1)

        y_np = labels.numpy()
        _, pred_nosol, pred_sol = _apply_thresholds(probs, tau, tau1_active)
        metrics = _compute_metrics(y_np, probs, pred_nosol, pred_sol)
        metrics["inference_ms_per_sample"] = round(ms_per_sample, 4)

        eval_results[split_name] = metrics
        append_gate_log(
            "pstar_finetune_head", split_name, metrics, tau, tau1_active,
            embedding_dim=input_dim,
        )
        print_gate_report(
            f"Step0b === PSTAR Fine-tuned Head — {split_name} ===",
            metrics, ms_per_sample=ms_per_sample,
        )

    out_path = config.get(
        "ragate_step0_eval_output",
        "Results/RAGate/log/RAGate_step0_eval.json",
    )
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nStep0b --> Results saved to {out_path}")
    if not has_head:
        print(
            "Step0b --> NOTE: This checkpoint was saved before head persistence was\n"
            "           added. Re-run Step 0 (ragate_embeddings_force_reextract=True)\n"
            "           to generate a new checkpoint with saved head weights."
        )

def train_mlp():
    from utils import set_seed
    set_seed()  
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

    input_dim = X_train.shape[1]
    print(f"MLP --> Train: {len(X_train):,} samples  |  Val: {len(X_val):,} samples")
    print(f"MLP --> Embedding dim: {input_dim} "
          f"({'mean+start+goal' if input_dim > 512 else 'mean only'})")

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

    optimizer   = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )

    print(f"MLP --> Loss: BCEWithLogitsLoss  pos_weight={pos_weight}"
          f"  ({'conservative — blocks more' if pos_weight < 1.0 else 'permissive — passes more' if pos_weight > 1.0 else 'standard BCE'})")

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
        net.train()
        running_loss = 0.0
        n_batches    = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = net(X_batch).squeeze(1)
            loss   = criterion(logits, y_batch)
            loss_val = float(loss.item())
            if not (loss_val < 0.5):             
                optimizer.state.clear()         
                optimizer.zero_grad()
                continue
            loss.backward()
            clip_val = config.get("gradient_clip_value", 1.0)
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            optimizer.step()
            running_loss += loss_val
            n_batches    += 1
        train_loss = running_loss / max(n_batches, 1)

        val_metrics, _ = _eval_on_split(
            net, X_val, y_val, device, tau, tau1_active
        )
        val_accuracy    = val_metrics["accuracy"]
        val_specificity = val_metrics["specificity"]
        val_fn_rate     = val_metrics["fn_rate"]

        if val_accuracy > best_val_f1:
            best_val_f1  = val_accuracy
            best_val_rec = val_specificity
            best_epoch   = epoch
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy":         val_accuracy,
                "val_specificity":      val_specificity,
                "pos_weight":           pos_weight,
                "hidden_layers":        hidden_layers,
                "arch_str":             net.arch_str,
                "input_dim":            input_dim,
            }, model_path)

        training_rows.append({
            "epoch":            epoch,
            "train_loss":       round(train_loss,       6),
            "pos_weight":       pos_weight,
            "val_accuracy":     round(val_accuracy,     4),
            "val_specificity":  round(val_specificity,  4),
            "val_fn_rate":      round(val_fn_rate,       4),
        })

        print(f"--> MLP Ep{epoch}/{n_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"pos_weight: {pos_weight} | "
              f"Val Accuracy (RGSR): {val_accuracy:.4f} | "
              f"Val Specificity: {val_specificity:.4f} | "
              f"Val FN Rate: {val_fn_rate:.4f}")

    log_path = config["ragate_mlp_training_log"]
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".",
                exist_ok=True)
    fieldnames = ["epoch", "train_loss", "pos_weight",
                  "val_accuracy", "val_specificity", "val_fn_rate"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_rows)

    print(f"\n--> MLP Classifier - Training complete")
    print(f"--> MLP Classifier - Architecture: {net.arch_str}")
    print(f"--> MLP Classifier - Best Val SGR: {best_val_f1:.4f} at Ep{best_epoch} "
          f"(first epoch kept on ties)")
    print(f"--> MLP Classifier - Checkpoint saved to {model_path}")

def evaluate_mlp():
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
          f"Val Accuracy (RGSR)={checkpoint.get('val_accuracy', checkpoint.get('val_sgr', checkpoint.get('val_f1', '?')))}  "
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
        print_gate_report(f"MLP Classifier === Evaluation: {split_name} ===",
                          metrics, ms_per_sample=metrics['inference_ms_per_sample'])
    eval_path = config["ragate_mlp_eval_output"]
    os.makedirs(os.path.dirname(eval_path) if os.path.dirname(eval_path) else ".",
                exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n--> MLP Classifier - Results saved to {eval_path}")
    sweep_path = config["ragate_mlp_sweep_output"]
    os.makedirs(os.path.dirname(sweep_path) if os.path.dirname(sweep_path) else ".",
                exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"--> MLP Classifier - Sweep saved to {sweep_path}")

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
