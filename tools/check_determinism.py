
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch                                   
from torch.utils.data import Subset            
from config import config                      
from utils import set_seed, make_generator, get_backbone_class   
from dataloader import get_dataloader, PathfindingDataset         
from direction_model import direction_model    
from training import train_epoch               

def _weights_hash(model):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def _one_run(train_subset, epochs, deterministic, device):
    set_seed(deterministic=deterministic, verbose=False)
    base = get_backbone_class()(
        num_unique_tokens=config["num_unique_tokens"], dim=config["dim"],
        num_layers=config["num_layers"], heads=config["heads"],
        sequence_len=config["sequence_len"], causal=config["causal"],
        ff_mult=config["ff_mult"], ff_dropout=config["ff_dropout"],
        attn_dropout=config["attn_dropout"]).to(device)
    model = direction_model(base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 **config.get("adam_params", {}))
    gen = make_generator(offset=0)
    losses = []
    for epoch in range(epochs):
        loader = get_dataloader(config["train_file"], config, "train_file_pickle",
                                base_dataset=train_subset, generator=gen)
        loss, _ = train_epoch(model, loader, optimizer, epoch, device, dataset_label="determinism check")
        losses.append(loss)
    return losses, _weights_hash(model)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.05, help="fraction of the training split (default 0.05)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--no-deterministic", action="store_true",
                    help="run with deterministic=False to see the difference")
    args = ap.parse_args()
    deterministic = not args.no_deterministic
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Architecture: {config.get('architecture', 'pstar')}  seed: {config.get('random_seed')}  "
          f"deterministic: {deterministic}  device: {device}")
    full = PathfindingDataset(config["train_file"], config, "train_file_pickle")
    n = max(config["batch"], int(len(full) * args.fraction))
    subset = Subset(full, list(range(min(n, len(full)))))
    print(f"Subset: {len(subset):,} of {len(full):,} training samples, {args.epochs} epoch(s)\n")
    runs = []
    for r in (1, 2):
        print(f"--- run {r} ---")
        runs.append(_one_run(subset, args.epochs, deterministic, device))
    (l1, h1), (l2, h2) = runs
    print("\nEpoch losses")
    for e, (a, b) in enumerate(zip(l1, l2), 1):
        print(f"  epoch {e}: run1 {a:.10f}  run2 {b:.10f}  {'same' if a == b else 'DIFFERENT'}")
    print(f"Weights hash: run1 {h1}  run2 {h2}")
    ok = (l1 == l2) and (h1 == h2)
    print("\nRESULT:", "PASS — same seed gives the same model" if ok
          else "FAIL — runs differ (check deterministic=True, CUBLAS_WORKSPACE_CONFIG and "
               "the [SEED]/warnings in the log)")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
