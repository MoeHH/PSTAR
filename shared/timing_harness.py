import csv
import heapq
import json
import os
import pickle
import platform
import time
from collections import deque
import numpy as np
import torch
from config import config
from dataloader import get_dataloader, get_sample_metadata
from inference import load_model_from_checkpoint
from utils import mask_logits, build_dynamic_mask, get_next_cell_info, _write_next_token
COMPONENTS = ("gate_ms", "rollout_ms", "unified_ms", "astar_ms", "bfs_ms")

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def _hardware_info(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return info

def _stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}

def _grid_from_context(context):
    n_rows, n_cols = (int(v) for v in config["grid_size"])
    obstacles, start, goal = set(), None, None
    for cell in context:
        cn = int(cell.get("Cell_Number", -1))
        if cn <= 0:
            continue
        rc = ((cn - 1) // n_cols, (cn - 1) % n_cols)
        st = int(cell.get("State_Status", 0))
        if st == int(config["status_obstacle"]):
            obstacles.add(rc)
        elif st == int(config["status_start"]):
            start = rc
        elif st == int(config["status_end"]):
            goal = rc
    return n_rows, n_cols, obstacles, start, goal

def astar(n_rows, n_cols, obstacles, start, goal):
    h = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])
    open_heap = [(h(start), 0, start)]
    g_best, closed, expansions = {start: 0}, set(), 0
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if cur == goal:
            return g + 1, expansions
        r, c = cur
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nb[0] < n_rows and 0 <= nb[1] < n_cols and nb not in obstacles \
                    and nb not in closed and g + 1 < g_best.get(nb, 1 << 30):
                g_best[nb] = g + 1
                heapq.heappush(open_heap, (g + 1 + h(nb), g + 1, nb))
    return None, expansions

def bfs_reachable(n_rows, n_cols, obstacles, start, goal):
    seen, q = {start}, deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        r, c = cur
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nb[0] < n_rows and 0 <= nb[1] < n_cols and nb not in obstacles and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return False

def _load_gate(device):
    kind = str(config.get("ragate_model", "xgboost")).strip().lower()
    if kind in ("lda", "xgboost"):
        path = config[f"ragate_{kind}_model_path"]
        with open(path, "rb") as f:
            clf = pickle.load(f)
        if kind == "xgboost":
            try:
                clf.get_booster().set_param("device", "cpu")
            except Exception:
                pass
        return kind, (lambda X: float(clf.predict_proba(X.cpu().numpy())[:, 1][0]))
    if kind == "mlp":
        from mlp_classifier import RAGateMLPNet
        ckpt = torch.load(config["ragate_mlp_model_path"], weights_only=False, map_location=device)
        net = RAGateMLPNet(input_dim=int(ckpt.get("input_dim", config["dim"])),
                           hidden_layers=list(ckpt.get("hidden_layers", config["ragate_mlp_hidden_layers"])),
                           dropout=float(config.get("ragate_mlp_dropout", 0.1))).to(device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        return kind, (lambda X: float(torch.sigmoid(net(X).squeeze()).item()))
    raise ValueError(f"Unknown ragate_model '{kind}'")

def _gate_prob(model, inp, predict):
    emb = model.get_context_embeddings(inp)
    parts = []
    if config.get("ragate_use_mean_emb", True) and "context_mean" in emb:
        parts.append(emb["context_mean"])
    if config.get("ragate_use_start_emb", False) and "start_emb" in emb:
        parts.append(emb["start_emb"])
    if config.get("ragate_use_goal_emb", False) and "goal_emb" in emb:
        parts.append(emb["goal_emb"])
    return predict(torch.cat(parts, dim=1))

def rollout_single(model, inp, context, device):
    ctx_len = int(config["context_matrix_length"])
    lat_len = int(config["foundroute_matrix_length"])
    pad = float(config["padding_value"])
    fidx_cell, fidx_x, fidx_y, fidx_st = (int(config[k]) for k in
                                          ("fidx_cell_num", "fidx_x", "fidx_y", "fidx_status"))

    cur = inp.clone()
    ssm = {c.get("Cell_Number", 0): c.get("State_Status", 0) for c in context}
    goal_cell, gx, gy = None, None, None
    for i in range(ctx_len):
        if int(cur[0, i, fidx_st].item()) == int(config["status_end"]):
            goal_cell = int(cur[0, i, fidx_cell].item())
            gx, gy = float(cur[0, i, fidx_x].item()), float(cur[0, i, fidx_y].item())
            break
    goal_xy = torch.tensor([[gx or 0.0, gy or 0.0]], device=device)
    start_cell = int(cur[0, ctx_len, fidx_cell].item())
    visited, route = {start_cell}, [start_cell]
    if lat_len > 1:
        cur[:, ctx_len + 1:, :] = pad

    forward_passes = 0
    for step in range(lat_len):
        cell = int(cur[0, ctx_len + step, fidx_cell].item())
        mvec = build_dynamic_mask(cell, config["grid_size"], ssm, visited)
        if not any(mvec):
            return False, route, forward_passes                       # stuck_masked
        logits = model.base_model(cur)
        forward_passes += 1
        ml = mask_logits(logits[:, step:step + 1, :], cur[:, ctx_len + step, fidx_cell].unsqueeze(1),
                         torch.tensor([mvec], dtype=torch.float32, device=device).unsqueeze(1))
        direction = int(ml.argmax(dim=-1)[0, 0].item())
        nxt = get_next_cell_info(cell, direction, config["grid_size"], ssm)
        if nxt is None:
            return False, route, forward_passes
        next_cell, x, y, st = nxt
        visited.add(int(next_cell))
        route.append(int(next_cell))
        if goal_cell is not None and int(next_cell) == goal_cell:
            return True, route, forward_passes
        if step < lat_len - 1:
            _write_next_token(cur, 0, slot=ctx_len + step + 1, next_cell=next_cell, x=x, y=y,
                              state_status=st, goal_cells_xy=goal_xy, context_len=ctx_len)
    return False, route, forward_passes                                # horizon

def _select_bins(dataset):
    pad = config["padding_value"]
    lengths = []
    for idx in range(len(dataset)):
        fr = dataset[idx][4]
        lengths.append(sum(1 for c in fr if c.get("Cell_Number", pad) != pad))
    lengths = np.asarray(lengths)
    pcts = list(config.get("timing_percentiles", [10, 50, 99]))
    names = list(config.get("timing_bin_names", [f"P{p}" for p in pcts]))
    k = int(config.get("timing_samples_per_bin", 20))
    bins, used = {}, set()
    for name, p in zip(names, pcts):
        target = float(np.percentile(lengths, p))
        order = sorted(range(len(lengths)), key=lambda i: (abs(lengths[i] - target), i))
        chosen = [i for i in order if i not in used][:k]
        used.update(chosen)
        bins[name] = {"percentile": p, "target_route_cells": target,
                      "indices": chosen,
                      "route_cells_range": [int(lengths[chosen].min()), int(lengths[chosen].max())]}
    dist = {"min": int(lengths.min()), "max": int(lengths.max()),
            **{f"P{p}": float(np.percentile(lengths, p)) for p in pcts}}
    return bins, dist

def timing_main():
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    out_dir = config["timing_results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    warm, runs = int(config.get("timing_warmup_runs", 10)), int(config.get("timing_runs", 50))
    tau = float(config["ragate_model_tau1"])
    tau_active = bool(config.get("ragate_model_tau1_active", True))

    model = load_model_from_checkpoint(
        os.path.join(config["checkpoint_dir"], config.get("inference_model_name", config["ce_best_model_name"])),
        device)
    model.eval()

    use_gate = bool(config.get("timing_include_gate", True))
    gate_kind, predict = (None, None)
    if use_gate:
        try:
            gate_kind, predict = _load_gate(device)
        except (FileNotFoundError, OSError) as e:
            print(f"TIMING --> RAGate classifier unavailable ({e}); gate/unified not timed.")
            use_gate = False

    dataset = get_dataloader(config["test_file"], config, "test_file_pickle", shuffle=False).dataset
    bins, length_dist = _select_bins(dataset)
    print(f"TIMING --> grid {config['grid_size']}  route-length distribution {length_dist}")

    per_run_path = os.path.join(out_dir, "timing_per_run.csv")
    per_run_f = open(per_run_path, "w", newline="")
    per_run = csv.writer(per_run_f)
    per_run.writerow(["bin", "dataset_index", "run", "component", "ms"])

    sample_rows, bin_values, peak_mem = [], {}, {c: 0 for c in COMPONENTS}

    with torch.no_grad():
        for bin_name, b in bins.items():
            for ds_idx in b["indices"]:
                inp, _, _, context, found_routes = dataset[ds_idx]
                inp = inp.unsqueeze(0).to(device)
                n_rows, n_cols, obs, s_rc, g_rc = _grid_from_context(context)
                true_len = sum(1 for c in found_routes if c.get("Cell_Number", -1) != config["padding_value"])

                def run_gate():
                    return _gate_prob(model, inp, predict)
                def run_rollout():
                    return rollout_single(model, inp, context, device)
                def run_unified():
                    p = run_gate()
                    return run_rollout() if (not tau_active or p >= tau) else None
                comps = {"rollout_ms": run_rollout,
                         "astar_ms": lambda: astar(n_rows, n_cols, obs, s_rc, g_rc),
                         "bfs_ms": lambda: bfs_reachable(n_rows, n_cols, obs, s_rc, g_rc)}
                if use_gate:
                    comps.update({"gate_ms": run_gate, "unified_ms": run_unified})
                reached, route, fwd = run_rollout()
                astar_len, expansions = astar(n_rows, n_cols, obs, s_rc, g_rc)
                prob = run_gate() if use_gate else None
                row = dict(get_sample_metadata(dataset, ds_idx), bin=bin_name, dataset_index=ds_idx,
                           true_route_cells=true_len, astar_route_cells=astar_len,
                           pstar_success=int(reached), pstar_route_cells=len(route),
                           pstar_exact_optimal=int(reached and len(route) == true_len),
                           pstar_forward_passes=fwd, astar_expansions=expansions,
                           gate_prob=None if prob is None else round(prob, 6),
                           gate_admits=None if prob is None else int((not tau_active) or prob >= tau))
                for comp, fn in comps.items():
                    for _ in range(warm):
                        fn()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    times = []
                    for r in range(runs):
                        _sync(device)
                        t0 = time.perf_counter()
                        fn()
                        _sync(device)
                        ms = (time.perf_counter() - t0) * 1000.0
                        times.append(ms)
                        per_run.writerow([bin_name, ds_idx, r, comp, f"{ms:.5f}"])
                    if device.type == "cuda" and comp in ("gate_ms", "rollout_ms", "unified_ms"):
                        peak_mem[comp] = max(peak_mem[comp], torch.cuda.max_memory_allocated())
                    st = _stats(times)
                    row.update({f"{comp}_mean": round(st["mean"], 5), f"{comp}_std": round(st["std"], 5),
                                f"{comp}_median": round(st["median"], 5), f"{comp}_p95": round(st["p95"], 5)})
                    bin_values.setdefault(bin_name, {}).setdefault(comp, []).extend(times)
                sample_rows.append(row)
                print(f"TIMING --> [{bin_name}] idx={ds_idx} cells={true_len} "
                      f"fwd={fwd} A*exp={expansions} rollout={row['rollout_ms_mean']:.2f}ms "
                      f"A*={row['astar_ms_mean']:.3f}ms")
    per_run_f.close()
    per_sample_path = os.path.join(out_dir, "timing_per_sample.csv")
    cols = sorted({k for r in sample_rows for k in r})
    with open(per_sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sample_rows)
    def _bin_summary(rows, values):
        out = {comp: _stats(v) for comp, v in values.items()}
        for key in ("pstar_forward_passes", "astar_expansions", "true_route_cells"):
            out[key] = _stats([r[key] for r in rows])
        out["pstar_success_rate_pct"] = 100.0 * np.mean([r["pstar_success"] for r in rows])
        out["ms_per_forward_pass"] = _stats([r["rollout_ms_mean"] / max(r["pstar_forward_passes"], 1)
                                             for r in rows])
        return out
    summary = {
        "grid_size": list(config["grid_size"]),
        "context_matrix_length": int(config["context_matrix_length"]),
        "foundroute_matrix_length": int(config["foundroute_matrix_length"]),
        "sequence_len": int(config["sequence_len"]),
        "model": {"dim": config["dim"], "num_layers": config["num_layers"], "heads": config["heads"],
                  "parameters": int(sum(p.numel() for p in model.parameters())),
                  "checkpoint": config.get("inference_model_name")},
        "gate": {"classifier": gate_kind, "tau": tau, "active": tau_active, "timed": use_gate},
        "protocol": {"batch_size": 1, "warmup_runs": warm, "timed_runs": runs,
                     "samples_per_bin": int(config.get("timing_samples_per_bin", 20)),
                     "clock": "time.perf_counter with torch.cuda.synchronize",
                     "astar_bfs_device": "CPU (pure Python)",
                     "note": "Rollout recomputes the full sequence each step (no KV/context cache)."},
        "hardware": _hardware_info(device),
        "route_length_distribution_cells": length_dist,
        "bins": {name: {"percentile": b["percentile"], "target_route_cells": b["target_route_cells"],
                        "route_cells_range": b["route_cells_range"],
                        **_bin_summary([r for r in sample_rows if r["bin"] == name], bin_values.get(name, {}))}
                 for name, b in bins.items()},
        "peak_gpu_memory_mb": {c: round(v / 2 ** 20, 2) for c, v in peak_mem.items() if v},
        "files": {"per_run": per_run_path, "per_sample": per_sample_path},
    }
    summary_path = os.path.join(out_dir, "timing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nTIMING --> mean ± std (ms) per bin")
    for name, b in summary["bins"].items():
        line = "  ".join(f"{c[:-3]}={b[c]['mean']:.3f}±{b[c]['std']:.3f}"
                         for c in COMPONENTS if c in b and b[c].get("n"))
        print(f"  {name:7s} fwd={b['pstar_forward_passes']['mean']:.1f} "
              f"A*exp={b['astar_expansions']['mean']:.1f}  {line}")
    print(f"TIMING --> summary: {summary_path}")
    return summary
