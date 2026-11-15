import csv
import json
import os
import platform
import shutil
import socket
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STAGE_ORDER = [
    "PSTAR_training", "inference",
    "RAGate_embedding_extraction", "RAGate_LDA_train", "RAGate_LDA_eval",
    "RAGate_MLP_train", "RAGate_MLP_eval", "RAGate_XGBoost_train", "RAGate_XGBoost_eval",
    "RAGate_compare", "unified_framework", "timing_benchmark",
]

def _report_dir(config):
    return os.path.join(config["results_dir"].rstrip("/"), "report")

def _arch(config):
    return str(config.get("architecture", "pstar"))

def _rfile(config, name):
    return f"{_arch(config)}_{name}"

def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return None

def record_stage(config, stage, seconds):
    """Add one stage duration to timing.json (latest run of a stage wins)."""
    d = _report_dir(config)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, _rfile(config, "timing.json"))
    data = _load_json(path) or {"stages": {}}
    prev = data["stages"].get(stage, {})
    data["stages"][stage] = {
        "seconds": round(float(seconds), 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": int(prev.get("runs", 0)) + 1,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

class StageTimer:
    def __init__(self, config, stage):
        self.config, self.stage = config, stage

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:                       # only completed stages count
            record_stage(self.config, self.stage, time.time() - self.t)
        return False

def _fmt(sec):
    if sec is None:
        return "n/a"
    sec = int(round(sec))
    return f"{sec // 3600}h {(sec % 3600) // 60:02d}m {sec % 60:02d}s"

def _machine():
    info = {"hostname": socket.gethostname(), "python": platform.python_version(),
            "cores": os.cpu_count(), "cpu": platform.processor() or platform.machine()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int(f.readline().split()[1]) / 2 ** 20)
    except OSError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        info["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    return info

def _training(config):
    out = {}
    log = os.path.join(config["results_dir"], "log", "evaluateroutes_log.csv")
    if os.path.exists(log):
        with open(log, newline="") as f:
            rows = list(csv.DictReader(f))
        train = [r for r in rows if r.get("dataset") == "train"
                 and r.get("loss_source") == "train_loop_epoch_avg"]
        val = [r for r in rows if r.get("dataset") == "val"
               and r.get("loss_source") == "route_eval_epoch_avg"]
        out["epochs_completed"] = len({r.get("epoch") for r in train}) or None
        if val:
            def f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None
            sr = [(int(float(r["epoch"])), f(r.get("success_rate"))) for r in val if f(r.get("success_rate")) is not None]
            if sr:
                out["final_val_success_rate"] = sr[-1][1]
                best = max(sr, key=lambda t: t[1])
                out["best_val_success_rate"] = {"epoch": best[0], "value": best[1]}
            out["final_val_loss"] = f(val[-1].get("loss_total"))
        if train:
            out["final_train_loss"] = float(train[-1].get("loss_total") or 0) or None
    ckpts = {}
    try:
        import torch
        for name in (config.get("ce_best_model_name", "ce_best_model.pt"),
                     config.get("best_model_name", "best_model.pt")):
            p = os.path.join(config["checkpoint_dir"], name)
            if os.path.exists(p):
                c = torch.load(p, map_location="cpu", weights_only=False)
                ckpts[name] = {"epoch": int(c.get("epoch", -1)) + 1,
                               "val_success_rate": c.get("val_success_rate"),
                               "val_loss": c.get("validating_loss")}
    except Exception as e:
        ckpts["error"] = f"{type(e).__name__}: {e}"
    out["checkpoints"] = ckpts
    out["selection_rule"] = (f"{config.get('ce_best_model_name', 'ce_best_model.pt')} = highest "
                             f"validation success rate (used for inference); "
                             f"{config.get('best_model_name', 'best_model.pt')} = lowest validation loss")
    return out

def _test(config):
    s = _load_json(os.path.join(config["results_dir"], "log", "test_summary.json"))
    if not s:
        return None
    return {k: s.get(k) for k in ("overall", "route_types", "failures_by_route_type",
                                  "by_group", "num_templates")}

def _leakage(config):
    out = {}
    for split in ("test", "val"):
        s = _load_json(os.path.join(config["results_dir"], "log", f"split_leakage_{split}.json"))
        if s:
            out[split] = {k: s.get(k) for k in ("eval_samples", "eval_overlap_counts",
                                                "eval_overlap_pct", "results_by_overlap")}
    return out or None

def _ragate(config):
    s = _load_json(config.get("ragate_comparison_output"))
    return s

def _unified(config):
    s = _load_json(config.get("inference_classifier_results_output"))
    if not s:
        return None
    return {k: s.get(k) for k in ("classifier", "tau", "gate_metrics", "gate_threshold_free_metrics",
                                  "end_to_end", "rollout") if k in s} or s

def _timing_benchmark(config):
    s = _load_json(os.path.join(config.get("timing_results_dir", ""), "timing_summary.json"))
    if not s:
        return None
    bins = {}
    for name, b in (s.get("bins") or {}).items():
        bins[name] = {k: (b.get(k) or {}).get("mean") for k in
                      ("gate_ms", "rollout_ms", "unified_ms", "astar_ms", "bfs_ms",
                       "pstar_forward_passes", "astar_expansions", "true_route_cells")}
    return {"bins": bins, "hardware": s.get("hardware"), "protocol": s.get("protocol"),
            "peak_gpu_memory_mb": s.get("peak_gpu_memory_mb")}

def write_report(config):
    rdir = _report_dir(config)
    os.makedirs(rdir, exist_ok=True)
    timing = _load_json(os.path.join(rdir, _rfile(config, "timing.json"))) or {"stages": {}}
    tag = os.path.basename(config["results_dir"].rstrip("/"))

    report = {
        "session": tag,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": _machine(),
        "settings": {k: config.get(k) for k in (
            "grid_size", "foundroute_matrix_length", "sequence_len", "split_mode",
            "epochs", "batch", "optimizer", "learning_rate", "dim", "num_layers", "heads",
            "architecture", "random_seed", "deterministic", "feature_size",
            "goal_distance_weight", "goal_proximity_weight", "goal_obstacle_weight",
            "inference_model_name", "ragate_model", "ragate_model_tau1", "ragate_pstar_frozen",
            "train_file", "test_file")},
        "timing": timing,
        "training": _training(config),
        "test": _test(config),
        "leakage": _leakage(config),
        "ragate_comparison": _ragate(config),
        "unified": _unified(config),
        "timing_benchmark": _timing_benchmark(config),
    }
    with open(os.path.join(rdir, _rfile(config, "report.json")), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Copy of the config.py this run used, so every result can be traced back.
    cfg_src = os.path.join(ROOT, "config.py")
    if os.path.exists(cfg_src):
        shutil.copy(cfg_src, os.path.join(rdir, _rfile(config, "config_used.py")))

    copies = [os.path.join(config["results_dir"], "log", n) for n in
              ("test_summary.json", "split_leakage_test.json", "split_leakage_val.json")]
    copies += [config.get("ragate_comparison_output"), config.get("inference_classifier_results_output"),
               os.path.join(config.get("timing_results_dir", ""), "timing_summary.json")]
    for p in copies:
        if p and os.path.exists(p):
            shutil.copy(p, rdir)

    m, st = report["machine"], report["settings"]
    L = [f"# {_arch(config)} run report — {tag}", "",
         f"Created {report['created']} on {m.get('hostname')} — {m.get('cpu')}, {m.get('cores')} cores, "
         f"{m.get('ram_gb', '?')} GB RAM; GPU {m.get('gpu', 'none')} "
         f"({m.get('gpu_memory_gb', '?')} GB); torch {m.get('torch')}, CUDA {m.get('cuda')}, "
         f"cuDNN {m.get('cudnn')}", "",
         "## Settings",
         f"grid {st['grid_size']}, latent {st['foundroute_matrix_length']}, sequence {st['sequence_len']}, "
         f"split {st['split_mode']}; epochs {st['epochs']}, batch {st['batch']}, "
         f"{st['optimizer']} lr {st['learning_rate']}, dim {st['dim']}, layers {st['num_layers']}, "
         f"heads {st['heads']}, architecture {st['architecture']}, seed {st['random_seed']}, "
         f"deterministic {st['deterministic']} (cuBLAS {m.get('cublas_workspace_config')}), "
         f"goal_distance_weight {st['goal_distance_weight']}", "",
         "## Timing (wall clock)", "", "| Stage | Duration | Finished | Runs |", "|---|---|---|---|"]
    stages = timing.get("stages", {})
    total = 0.0
    for s in STAGE_ORDER + sorted(k for k in stages if k not in STAGE_ORDER):
        if s in stages:
            v = stages[s]
            total += v["seconds"]
            L.append(f"| {s} | {_fmt(v['seconds'])} | {v['finished']} | {v['runs']} |")
    L.append(f"| **Total** | **{_fmt(total)}** | | |")
    tr = report["training"]
    if tr.get("epochs_completed") and "PSTAR_training" in stages:
        L.append(f"\nTraining: {tr['epochs_completed']} epochs, "
                 f"{_fmt(stages['PSTAR_training']['seconds'] / tr['epochs_completed'])} per epoch on average.")

    L += ["", "## Training"]
    for name, c in (tr.get("checkpoints") or {}).items():
        if isinstance(c, dict):
            L.append(f"- `{name}`: epoch {c.get('epoch')}, val success {c.get('val_success_rate')}, "
                     f"val loss {c.get('val_loss')}")
    if tr.get("best_val_success_rate"):
        L.append(f"- Best val success {tr['best_val_success_rate']['value']}% at epoch "
                 f"{tr['best_val_success_rate']['epoch']}; final val success {tr.get('final_val_success_rate')}%")
    L.append(f"- {tr['selection_rule']}")

    t = report["test"]
    if t and t.get("overall"):
        o = t["overall"]
        L += ["", "## Test results",
              f"- Success: {o.get('success')}/{o.get('n')} = {o.get('success_rate_pct')}% "
              f"(95% CI {o.get('success_rate_ci95_pct')})",
              f"- Exact-optimal: {o.get('exact_optimal')} = {o.get('exact_optimal_rate_pct')}% "
              f"(95% CI {o.get('exact_optimal_rate_ci95_pct')}); mean excess steps "
              f"{o.get('mean_excess_steps_success')}", "",
              "| Route type | Count | % of routes |", "|---|---|---|"]
        n = max(o.get("n") or 1, 1)
        for k, v in (t.get("route_types") or {}).items():
            L.append(f"| {k} | {v} | {100.0 * v / n:.2f}% |")
        if t.get("by_group"):
            L += ["", "| Template group | n | Success % | Exact-optimal % |", "|---|---|---|---|"]
            for g, b in t["by_group"].items():
                L.append(f"| {g} | {b.get('n')} | {b.get('success_rate_pct')} | {b.get('exact_optimal_rate_pct')} |")

    lk = report["leakage"]
    if lk:
        L += ["", "## Split leakage"]
        for split, v in lk.items():
            L.append(f"- {split}: {v.get('eval_overlap_pct')}")
            for g, r in (v.get("results_by_overlap") or {}).items():
                L.append(f"  - {g}: n={r.get('n')}, success {r.get('success_rate_pct')}%, "
                         f"exact-optimal {r.get('exact_optimal_rate_pct')}%")

    rg = report["ragate_comparison"]
    if rg:
        L += ["", "## RAGate", f"See `{os.path.basename(config.get('ragate_comparison_output', ''))}` "
              f"(recommendation: {rg.get('recommendation', rg.get('recommended', 'n/a'))})"]
    un = report["unified"]
    if un:
        e2e = (un.get("end_to_end") or {})
        g = (un.get("gate_metrics") or {})
        L += ["", "## Unified framework",
              f"- Outcomes: {e2e.get('outcome_counts')}",
              f"- End-to-end solve rate {e2e.get('end_to_end_solve_rate_pct')}%, exact-optimal "
              f"{e2e.get('end_to_end_exact_optimal_rate_pct')}%",
              f"- Gate: Recall {g.get('recall')}, Specificity {g.get('specificity')}, "
              f"FN Rate {g.get('fn_rate')}, Accuracy/RGSR {g.get('accuracy')}, "
              f"Precision {g.get('precision')}, F1 {g.get('f1')}, MCC {g.get('mcc')}"]
    tb = report["timing_benchmark"]
    if tb:
        L += ["", "## Timing benchmark (batch 1, mean ms)", "",
              "| Bin | route cells | PSTAR fwd passes | rollout | gate | unified | A* | A* expansions | BFS |",
              "|---|---|---|---|---|---|---|---|---|"]
        r = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
        for name, b in tb["bins"].items():
            L.append(f"| {name} | {r(b['true_route_cells'], 1)} | {r(b['pstar_forward_passes'], 1)} | "
                     f"{r(b['rollout_ms'])} | {r(b['gate_ms'])} | {r(b['unified_ms'])} | {r(b['astar_ms'])} | "
                     f"{r(b['astar_expansions'], 1)} | {r(b['bfs_ms'])} |")
    with open(os.path.join(rdir, _rfile(config, "report.md")), "w") as f:
        f.write("\n".join(L) + "\n")

    _res = config["results_dir"].rstrip("/")
    bundle = os.path.join(os.path.dirname(_res), f"{_arch(config)}_{os.path.basename(_res)}_report.tgz")
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(rdir, arcname=f"{_arch(config)}_{tag}_report")
    print(f"\n  [REPORT] {os.path.join(rdir, _rfile(config, 'report.md'))}")
    print(f"  [REPORT] bundle: {bundle}")
    return report

if __name__ == "__main__":
    from config import config as _cfg
    write_report(_cfg)
