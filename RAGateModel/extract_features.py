
import os
import sys
import time

import torch
import numpy as np

# ── Path setup so this script works when run from the project root ────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from inference import load_model_from_checkpoint

# ── Random seed ───────────────────────────────────────────────────────────────
_SEED = config.get("random_seed", 42)
torch.manual_seed(_SEED)
np.random.seed(_SEED)


# =============================================================================
# HELPERS
# =============================================================================

def _label_counts(labels):
    """Return (n_sol, n_nosol) from a list of int labels."""
    n_sol   = sum(1 for lb in labels if lb == 1)
    n_nosol = sum(1 for lb in labels if lb == 0)
    return n_sol, n_nosol


def _process_split(split_name, pt_path, output_path, model, device,
                   use_mean, use_start, use_goal):
    """
    Load one RAGate dataset split, run get_context_embeddings() in batches,
    and save the resulting embedding tensors + labels to output_path.

    Skips processing if output_path already exists AND
    config["ragate_embeddings_force_reextract"] is False (default).
    Set force_reextract=True to overwrite existing embedding files.
    """
    from torch.utils.data import Dataset, DataLoader

    # ── Skip if already extracted ─────────────────────────────────────────────
    force = bool(config.get("ragate_embeddings_force_reextract", False))
    if os.path.exists(output_path) and not force:
        print(f"--> Embedding file exists — skipping {split_name}: {output_path}")
        print(f"    (set ragate_embeddings_force_reextract=True to overwrite)")
        return

    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"EMBED --> RAGate dataset not found: {pt_path}\n"
            f"        Generate it with the A* dataset-generation repo and copy Datasets/RAGate/ here."
        )

    # ── Load Phase 1 samples ─────────────────────────────────────────────────
    print(f"--> Processing {split_name}: loading {pt_path} ...", flush=True)
    samples = torch.load(pt_path, weights_only=False)
    n_samples = len(samples)
    batch_size = int(config.get("ragate_embeddings_batch_size", 512))
    print(f"--> Processing {split_name}: {n_samples:,} samples  "
          f"(batch_size={batch_size})", flush=True)

    # ── Inline Dataset — stacks input_tensors and labels from sample dicts ───
    # Each sample dict has "input_tensor" [175, feature_size] and "label" int.
    # We only need these two fields for embedding extraction.
    class _RAGateDataset(Dataset):
        def __init__(self, samples):
            self._samples = samples
        def __len__(self):
            return len(self._samples)
        def __getitem__(self, idx):
            s = self._samples[idx]
            return s["input_tensor"], int(s["label"])

    loader = DataLoader(
        _RAGateDataset(samples),
        batch_size=batch_size,
        shuffle=False,          # preserve order so labels align with embeddings
        num_workers=0,          # samples already in RAM — no I/O benefit from workers
        pin_memory=(device.type == "cuda"),
    )

    # ── Accumulators ─────────────────────────────────────────────────────────
    all_mean_embs  = [] if use_mean  else None
    all_start_embs = [] if use_start else None
    all_goal_embs  = [] if use_goal  else None
    all_labels     = []

    n_batches   = len(loader)
    t_start     = time.perf_counter()
    samples_done = 0

    with torch.no_grad():
        for batch_idx, (inp_batch, label_batch) in enumerate(loader):

            inp_batch = inp_batch.to(device)

            result = model.get_context_embeddings(inp_batch)

            if use_mean:
                all_mean_embs.append(result["context_mean"].cpu())
            if use_start:
                all_start_embs.append(result["start_emb"].cpu())     # [B, 512]
            if use_goal:
                all_goal_embs.append(result["goal_emb"].cpu())       # [B, 512]

            all_labels.extend(label_batch.tolist())
            samples_done += len(label_batch)

            if (batch_idx + 1) % max(1, n_batches // 10) == 0 or \
               (batch_idx + 1) == n_batches:
                elapsed = time.perf_counter() - t_start
                print(f"  {split_name}: {samples_done:>7,}/{n_samples:,}  "
                      f"({samples_done/n_samples*100:.0f}%)  {elapsed:.1f}s",
                      end="\r", flush=True)

    print(flush=True)  

    elapsed = time.perf_counter() - t_start

    # ── Concatenate tensors ───────────────────────────────────────────────────
    y = torch.tensor(all_labels, dtype=torch.long)   # [N]

    save_dict = {
        "y":      y,
        "labels": all_labels,
    }

    if use_mean:
        X_mean = torch.cat(all_mean_embs, dim=0)     # [N, 512]
        save_dict["X"] = X_mean

    if use_start:
        X_start = torch.cat(all_start_embs, dim=0)   # [N, 512]
        save_dict["X_start"] = X_start

    if use_goal:
        X_goal = torch.cat(all_goal_embs, dim=0)     # [N, 512]
        save_dict["X_goal"] = X_goal

    # ── Build combined embedding X_combined ───────────────────────────────────
    parts = []
    if use_mean  and "X"       in save_dict: parts.append(save_dict["X"])
    if use_start and "X_start" in save_dict: parts.append(save_dict["X_start"])
    if use_goal  and "X_goal"  in save_dict: parts.append(save_dict["X_goal"])
    if parts:
        save_dict["X_combined"] = torch.cat(parts, dim=1)
        combined_dim = save_dict["X_combined"].shape[1]
    else:
        combined_dim = 0

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True)
    torch.save(save_dict, output_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    n_sol, n_nosol = _label_counts(all_labels)
    x_shape_str      = str(tuple(save_dict["X"].shape))          if use_mean  else "not extracted"
    xcomb_shape_str  = str(tuple(save_dict["X_combined"].shape)) if parts     else "n/a"
    print(f"--> Embed Context - {split_name} — shape: X={x_shape_str}  X_combined={xcomb_shape_str}  y={tuple(y.shape)}")
    print(f"--> Embed Context - {split_name} — label dist: sol={n_sol:,}  nosol={n_nosol:,}")
    print(f"--> Embed Context - {split_name} — time: {elapsed:.1f}s")
    print(f"--> Embed Context - {split_name} — saved to {output_path}")
    print()


# =============================================================================
# VERIFICATION
# =============================================================================

def _verify_embedding_files(splits):

    print("── Verification ──────────────────────────────────────────────")
    all_ok = True
    for split_name, output_path, expected_n in splits:
        if not os.path.exists(output_path):
            print(f"  [FAIL] {split_name}: file not found — {output_path}")
            all_ok = False
            continue

        data = torch.load(output_path, weights_only=False)
        y    = data["y"]
        n    = y.shape[0]

        # Shape check
        if n != expected_n:
            print(f"  [FAIL] {split_name}: y has {n} rows, expected {expected_n}")
            all_ok = False

        # X shape (if present)
        if "X" in data:
            X = data["X"]
            if X.shape[0] != n:
                print(f"  [FAIL] {split_name}: X rows ({X.shape[0]}) != y rows ({n})")
                all_ok = False
            # Expected X dim accounts for all_layers mode
            all_layers  = bool(config.get("ragate_embedding_all_layers", False))
            n_layers    = int(config.get("num_layers", 1))
            layer_mult  = n_layers if all_layers else 1
            expected_x_dim = config["dim"] * layer_mult
            if X.shape[1] != expected_x_dim:
                print(f"  [FAIL] {split_name}: X dim {X.shape[1]} != "
                      f"expected {expected_x_dim} "
                      f"(dim={config['dim']} × {'all '+str(n_layers)+' layers' if all_layers else 'last layer'})")
                all_ok = False

        # X_combined shape (if present — what classifiers actually use)
        if "X_combined" in data:
            X_comb = data["X_combined"]
            if X_comb.shape[0] != n:
                print(f"  [FAIL] {split_name}: X_combined rows ({X_comb.shape[0]}) != y rows ({n})")
                all_ok = False
            comb_dim    = X_comb.shape[1]
            all_layers  = bool(config.get("ragate_embedding_all_layers", False))
            n_layers    = int(config.get("num_layers", 1))
            layer_mult  = n_layers if all_layers else 1
            n_active    = sum([
                bool(config.get("ragate_use_mean_emb",  True)),
                bool(config.get("ragate_use_start_emb", False)),
                bool(config.get("ragate_use_goal_emb",  False)),
            ])
            # mean uses dim * layer_mult; start/goal always use dim (last layer)
            use_mean    = bool(config.get("ragate_use_mean_emb", True))
            use_start   = bool(config.get("ragate_use_start_emb", False))
            use_goal    = bool(config.get("ragate_use_goal_emb",  False))
            expected_comb_dim = (config["dim"] * layer_mult if use_mean else 0) + \
                                (config["dim"] if use_start else 0) + \
                                (config["dim"] if use_goal  else 0)
            if comb_dim != expected_comb_dim:
                print(f"  [WARN] {split_name}: X_combined dim={comb_dim} "
                      f"(expected {expected_comb_dim})")

        # Label balance check
        n_sol   = int((y == 1).sum().item())
        n_nosol = int((y == 0).sum().item())
        if n_sol != n_nosol:
            print(f"  [WARN] {split_name}: label imbalance — sol={n_sol}  nosol={n_nosol}")
        else:
            x_shape    = tuple(data["X"].shape)          if "X"         in data else "n/a"
            xcomb_shape= tuple(data["X_combined"].shape) if "X_combined" in data else "n/a"
            print(f"  [OK]   {split_name}: {n:,} samples  sol={n_sol:,}  nosol={n_nosol:,}  "
                  f"X={x_shape}  X_combined={xcomb_shape}")

    if all_ok:
        print("--> Embed Context - All embedding files verified OK")
    else:
        print("[WARNING] One or more embedding files failed verification.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Device setup ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)
    print(f"--> Device: {device}")

    # ── Embedding toggles ─────────────────────────────────────────────────────
    use_mean  = bool(config.get("ragate_use_mean_emb",  True))
    use_start = bool(config.get("ragate_use_start_emb", False))
    use_goal  = bool(config.get("ragate_use_goal_emb",  False))

    # ── Split definitions ─────────────────────────────────────────────────────
    splits_cfg = [
        ("train",
         config["ragate_train_pt"],
         config["ragate_embeddings_train"],
         160_000),
        ("val",
         config["ragate_val_pt"],
         config["ragate_embeddings_val"],
         20_000),
        ("test",
         config["ragate_test_pt"],
         config["ragate_embeddings_test"],
         20_000),
    ]

    force = bool(config.get("ragate_embeddings_force_reextract", False))

    all_exist = all(os.path.exists(out) for _, _, out, _ in splits_cfg)
    if all_exist and not force:
        print(f"--> All embedding files already exist — skipping extraction entirely.")
        print(f"    (set ragate_embeddings_force_reextract=True to overwrite)")
        print()
        # Run verification only
        verify_list = [(name, out, exp_n) for name, _, out, exp_n in splits_cfg]
        _verify_embedding_files(verify_list)
        return

    checkpoint_path = os.path.join(
        config["checkpoint_dir"],
        config["inference_model_name"]
    )
    model = load_model_from_checkpoint(checkpoint_path, device)

    # to freez and un freez 
    if config.get("ragate_pstar_frozen", True):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("--> PSTAR: FROZEN")
    else:
        model.train()
        for p in model.parameters():
            p.requires_grad = True
        print("--> PSTAR: UNFROZEN")

    print(f"--> PSTAR loaded from {checkpoint_path}")
    print(f"--> Embedding toggles: mean={use_mean}  start={use_start}  goal={use_goal}")
    print()

    # ── Process each split ────────────────────────────────────────────────────
    for split_name, pt_path, output_path, _ in splits_cfg:
        _process_split(
            split_name  = split_name,
            pt_path     = pt_path,
            output_path = output_path,
            model       = model,
            device      = device,
            use_mean    = use_mean,
            use_start   = use_start,
            use_goal    = use_goal,
        )

    # ── Final verification ────────────────────────────────────────────────────
    verify_list = [
        (name, out_path, exp_n)
        for name, _, out_path, exp_n in splits_cfg
    ]
    _verify_embedding_files(verify_list)


if __name__ == "__main__":
    main()
