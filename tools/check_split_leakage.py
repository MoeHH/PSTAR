import argparse
import csv
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "shared", "PSTARModel", "RAGateModel"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from config import config  # noqa: E402

def _iter_samples(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                except EOFError:
                    break
                if isinstance(obj, list):
                    yield from obj
                else:
                    yield obj
    elif path.endswith(".json"):
        with open(path) as f:
            yield from json.load(f)
    elif path.endswith(".pt"):
        import torch
        yield from torch.load(path, weights_only=False)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def _query_key(sample):
    st_obs = int(config["status_obstacle"])
    st_s, st_g = int(config["status_start"]), int(config["status_end"])
    pad = config["padding_value"]
    obstacles, start, goal = [], None, None
    ctx = sample.get("Context")
    if ctx is not None:
        cells = ((c.get("Cell_Number", pad), c.get("State_Status", 0)) for c in ctx)
    else:
        ctx_len = int(config["context_matrix_length"])
        fc, fs = int(config["fidx_cell_num"]), int(config["fidx_status"])
        rows = sample["input_tensor"][:ctx_len, [fc, fs]].tolist()
        cells = ((int(cn), int(st)) for cn, st in rows)
    for cn, st in cells:
        if cn == pad or cn <= 0:
            continue
        if st == st_obs:
            obstacles.append(int(cn))
        elif st == st_s:
            start = int(cn)
        elif st == st_g:
            goal = int(cn)
    return tuple(sorted(obstacles)), start, goal

def _resolve(split):
    for key in (f"{split}_file_pickle", f"{split}_file"):
        path = config.get(key)
        if path and os.path.exists(path):
            return path
    pt = config.get(f"{split}_file", "").rsplit(".", 1)[0] + ".pt"
    if os.path.exists(pt):
        return pt
    raise FileNotFoundError(f"No {split} file found (checked {split}_file_pickle / {split}_file).")

def _rate(rows, key):
    n = len(rows)
    k = sum(int(r.get(key, 0) or 0) for r in rows)
    return {"n": n, key: k, f"{key}_rate_pct": round(100.0 * k / n, 3) if n else None}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=None, help="training split file (default: from config)")
    ap.add_argument("--eval", default="test", choices=["val", "test"])
    ap.add_argument("--eval-file", default=None, help="evaluation split file (default: from config)")
    ap.add_argument("--per-sample", default=None, help="inference per-sample CSV to join")
    args = ap.parse_args()

    train_path = args.train or _resolve("train")
    eval_path = args.eval_file or _resolve(args.eval)
    print(f"Train : {train_path}\nEval  : {eval_path}  ({args.eval})")

    train_queries, train_layouts = set(), set()
    n_train = 0
    for s in _iter_samples(train_path):
        obs, a, b = _query_key(s)
        train_queries.add((obs, a, b))
        train_layouts.add(obs)
        n_train += 1

    flags, eval_queries = [], {}
    for i, s in enumerate(_iter_samples(eval_path)):
        obs, a, b = _query_key(s)
        if (obs, a, b) in train_queries:
            f = "same_query"
        elif (obs, b, a) in train_queries:
            f = "reverse_query"
        elif obs in train_layouts:
            f = "same_layout"
        else:
            f = "unseen"
        flags.append(f)
        eval_queries.setdefault((obs, a, b), []).append(i)

    n_eval = len(flags)
    counts = {k: flags.count(k) for k in ("same_query", "reverse_query", "same_layout", "unseen")}
    pct = {k: round(100.0 * v / n_eval, 3) if n_eval else None for k, v in counts.items()}
    dup = sum(len(v) for v in eval_queries.values() if len(v) > 1)
    report = {
        "grid_size": list(config["grid_size"]),
        "train_file": train_path, "eval_file": eval_path, "eval_split": args.eval,
        "train_samples": n_train, "train_unique_queries": len(train_queries),
        "train_unique_layouts": len(train_layouts),
        "eval_samples": n_eval, "eval_unique_queries": len(eval_queries),
        "eval_samples_sharing_a_query_within_eval": dup,
        "eval_overlap_counts": counts, "eval_overlap_pct": pct,
        "definitions": {
            "same_query": "same obstacles, start and goal as a training sample",
            "reverse_query": "same obstacles, start/goal swapped",
            "same_layout": "same obstacle layout, different endpoints",
            "unseen": "obstacle layout never seen in training"},
    }

    if args.per_sample:
        with open(args.per_sample, newline="") as f:
            rows = {int(r["dataset_index"]): r for r in csv.DictReader(f)}
        groups = {"leaked_query (same or reverse)": [], "same_layout_only": [], "unseen_layout": []}
        for i, flag in enumerate(flags):
            r = rows.get(i)
            if r is None:
                continue
            if flag in ("same_query", "reverse_query"):
                groups["leaked_query (same or reverse)"].append(r)
            elif flag == "same_layout":
                groups["same_layout_only"].append(r)
            else:
                groups["unseen_layout"].append(r)
        report["results_by_overlap"] = {
            g: {**_rate(rs, "success"), **{k: v for k, v in _rate(rs, "exact_optimal").items() if k != "n"}}
            for g, rs in groups.items()}

    out_dir = os.path.join(config["results_dir"], "log")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"split_leakage_{args.eval}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTrain: {n_train:,} samples, {len(train_queries):,} unique queries, "
          f"{len(train_layouts):,} unique layouts")
    print(f"Eval : {n_eval:,} samples, {len(eval_queries):,} unique queries "
          f"({dup:,} samples share a query with another eval sample)")
    for k in counts:
        print(f"  {k:<14}: {counts[k]:>8,}  ({pct[k]}%)")
    for g, v in report.get("results_by_overlap", {}).items():
        print(f"  {g:<32} n={v['n']:>7,}  success={v['success_rate_pct']}%  "
              f"exact-optimal={v['exact_optimal_rate_pct']}%")
    print(f"--> {out}")

if __name__ == "__main__":
    main()
