
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from config import config
from stuck_masked_subtypes import (classify_route, get_route_type_names, get_route_type_color_map, ROUTE_TYPE_SUCCESS,)

# Module-level helper (was duplicated as a local def in two functions)
def _to_float_or_none(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def detect_dataset_name(dataset_path):
    name = dataset_path.lower()
    if "train" in name:
        return "train"
    if "val" in name or "valid" in name:
        return "val"
    if "test" in name:
        return "test"
    return "unknown"

def classify_failure_reason(pred_route, true_route, goal_cell, context,
                            grid_size, max_path_len=None, term_reason=None):
    padding_value = config["padding_value"]
    if len(pred_route) > 0 and isinstance(pred_route[0], dict):
        pred_cells = [cell["Cell_Number"] for cell in pred_route
                      if cell.get("Cell_Number") != padding_value]
    else:
        pred_cells = [cell for cell in pred_route if cell != padding_value]
    state_status_map = {cell.get("Cell_Number", 0): cell.get("State_Status", 0)
                        for cell in (context or [])}
    route_type, _ = classify_route(pred_cells, goal_cell, grid_size, state_status_map, config)
    return route_type

def _write_results_row(csv_path, results):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_header = list(results.keys())
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_header)
            writer.writeheader()
            writer.writerow(results)
        return

    with open(csv_path, "r", newline="") as f:
        reader          = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        old_rows        = list(reader)

    if existing_header == new_header:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=new_header).writerow(results)
        return

    union_header = []
    seen = set()
    for h in existing_header + new_header:
        if h and h not in seen:
            union_header.append(h)
            seen.add(h)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=union_header)
        writer.writeheader()
        for row in old_rows:
            writer.writerow({k: row.get(k, "") for k in union_header})
        writer.writerow({k: results.get(k, "") for k in union_header})


def _append_jsonl(jsonl_path, results):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")


def _normalize_termination_reason(reason):
    """Canonical rollout termination signal: success | stuck_masked | horizon | other."""
    r = str(reason or "").strip().lower()
    if r in ("success", "goal_reached", "reached_goal"):
        return "success"
    if r in ("stuck_masked", "stuck_all_masked", "no_valid_actions", "masked_out"):
        return "stuck_masked"
    if r in ("max_length", "horizon", "hit_horizon", "max_steps_unreached"):
        return "horizon"
    return "other"


def evaluate_routes(predicted_routes,true_routes,goal_positions,epoch,dataset_name="unknown",
        total_samples=None,sample_indices=None,start_cells=None,run_id=None,contexts=None,termination_reasons=None,
        loss_total=None,loss_ce=None,loss_goal_delta=None,loss_goal_prox=None,loss_goal_obstacle=None
):
    dim_size     = config.get("dim")
    num_layers   = config.get("num_layers")
    num_heads    = config.get("heads")
    batch_size   = config.get("batch")
    total_epochs = config.get("epochs")
    results_dir  = config.get("results_dir", "results")
    padding_value = config.get("padding_value", -1)

    foundroute_len    = int(config.get("foundroute_matrix_length", 0))
    max_pred_path_len = foundroute_len          # horizon check: pred_len >= this
    max_path_len      = foundroute_len + 1      # passed to classify_failure_reason

    total_routes = len(predicted_routes)
    successful   = 0
    step_ratios  = []

    route_type_names  = get_route_type_names()
    route_type_counts = {name: 0 for name in route_type_names}
    route_type_colors = get_route_type_color_map()
    termination_counts = {"success": 0, "stuck_masked": 0, "horizon": 0, "other": 0}
    integrity_not_masked = 0   
    integrity_free_left  = 0      
    sub_diagnostics  = []         
    extra_steps          = []
    exact_match_steps    = 0
    fewer_steps_than_truth = 0
    hit_horizon_count    = 0
    hit_horizon_success_count = 0

    if run_id is None:
        run_id = "unknown"
    if true_routes is None:
        true_routes = []
    if goal_positions is None:
        goal_positions = []

    termination_reason_raw = []
    route_types            = []

    for idx in range(total_routes):
        pred  = predicted_routes[idx]
        truth = true_routes[idx]  if idx < len(true_routes)    else []
        goal  = goal_positions[idx] if idx < len(goal_positions) else padding_value

        # Normalise pred to plain int list
        if pred and isinstance(pred[0], dict):
            pred_cell_numbers = [c["Cell_Number"] for c in pred
                                 if c.get("Cell_Number") != padding_value]
        else:
            pred_cell_numbers = [c for c in pred if c != padding_value]

        # Normalise truth to plain int list
        if truth and isinstance(truth[0], dict):
            truth_cell_numbers = [c["Cell_Number"] for c in truth
                                  if c.get("Cell_Number") != padding_value]
        else:
            truth_cell_numbers = [c for c in truth if c != padding_value]

        pred_len  = len(pred_cell_numbers)
        truth_len = len(truth_cell_numbers)

        goal_cell     = int(goal) if goal != padding_value else padding_value
        pred_last_cell = pred_cell_numbers[-1] if pred_len > 0 else None
        hit_horizon = (max_pred_path_len > 0) and (pred_len >= max_pred_path_len)
        if hit_horizon:
            hit_horizon_count += 1

        # Termination signal from rollout
        term_norm = "other"
        if termination_reasons and idx < len(termination_reasons):
            term_norm = _normalize_termination_reason(termination_reasons[idx])
        termination_reason_raw.append(term_norm)

        # Ground-truth success check
        reached = (pred_len > 0) and (goal_cell != padding_value) and (pred_last_cell == goal_cell)

        if reached:
            successful += 1
            termination_counts[term_norm] = termination_counts.get(term_norm, 0) + 1
            route_type_counts[ROUTE_TYPE_SUCCESS] += 1
            route_types.append(ROUTE_TYPE_SUCCESS)

            if hit_horizon:
                hit_horizon_success_count += 1

            step_ratios.append(pred_len / max(truth_len, 1))
            extra_steps.append(pred_len - truth_len)
            if pred_len == truth_len:
                exact_match_steps += 1
            if pred_len < truth_len:
                fewer_steps_than_truth += 1
            continue

        # Failure path: one of the four stuck_masked subtypes
        context = contexts[idx] if contexts and idx < len(contexts) else []
        state_status_map = {cell.get("Cell_Number", 0): cell.get("State_Status", 0)
                            for cell in context}
        route_type, analysis = classify_route(
            pred_cell_numbers, goal_cell if goal_cell != padding_value else None,
            config["grid_size"], state_status_map, config)
        route_type_counts[route_type] = route_type_counts.get(route_type, 0) + 1
        route_types.append(route_type)
        termination_counts[term_norm] = termination_counts.get(term_norm, 0) + 1
        if term_norm != "stuck_masked":
            integrity_not_masked += 1
        if analysis is not None:
            sub_diagnostics.append(analysis)
            if analysis.n_free > 0:
                integrity_free_left += 1

    # Aggregate stats
    success_rate     = (successful / max(total_routes, 1)) * 100.0
    mean_step_ratio  = float(np.mean(step_ratios))  if step_ratios  else 0.0
    mean_extra_steps = float(np.mean(extra_steps))  if extra_steps  else 0.0
    hit_horizon_rate    = (hit_horizon_count / max(total_routes, 1)) * 100.0
    horizon_fail_count  = termination_counts.get("horizon", 0)
    horizon_success_rate = (
        hit_horizon_success_count * 100.0 / hit_horizon_count
        if hit_horizon_count > 0 else 0.0)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = {
        "run_id":        run_id,
        "timestamp":     timestamp,
        "dataset":       dataset_name,
        "epoch":         epoch,
        "dim_size":      dim_size,
        "num_layers":    num_layers,
        "num_heads":     num_heads,
        "batch_size":    batch_size,
        "total_epochs":  total_epochs,

        "total_routes":       total_routes,
        "successful_routes":  successful,
        "unsuccessful_routes": total_routes - successful,
        "success_rate":       round(success_rate, 2),

        "mean_step_ratio":       round(mean_step_ratio, 3),
        "mean_extra_steps":      round(mean_extra_steps, 3),
        "exact_match_steps":     exact_match_steps,
        "fewer_steps_than_truth": fewer_steps_than_truth,

        "hit_horizon_count":         hit_horizon_count,
        "hit_horizon_rate":          round(hit_horizon_rate, 2),
        "hit_horizon_success_count": hit_horizon_success_count,
        "hit_horizon_success_rate":  round(horizon_success_rate, 2),
        "hit_horizon_fail_count":    horizon_fail_count,

        "loss_source":        "route_eval_epoch_avg",
        "loss_total":         _to_float_or_none(loss_total),
        "loss_ce":            _to_float_or_none(loss_ce),
        "loss_goal_delta":    _to_float_or_none(loss_goal_delta),
        "loss_goal_prox":     _to_float_or_none(loss_goal_prox),
        "loss_goal_obstacle": _to_float_or_none(loss_goal_obstacle),

        "ce_temperature":       _to_float_or_none(config.get("ce_temperature")),
        "goal_distance_weight": _to_float_or_none(config.get("goal_distance_weight")),
        "goal_proximity_weight":_to_float_or_none(config.get("goal_proximity_weight")),
        "goal_obstacle_weight": _to_float_or_none(config.get("goal_obstacle_weight")),

        "termination_reason_raw":    json.dumps(termination_reason_raw),
        "route_types":               json.dumps(route_types),
    }

    results.update({
        **{f"route_{name}": route_type_counts.get(name, 0) for name in route_type_names},
        "integrity_fail_not_masked": integrity_not_masked,
        "integrity_fail_free_left":  integrity_free_left,
        "termination_counts": json.dumps(termination_counts),
    })

    if sub_diagnostics:
        results["sub_masked_avg_n_boundary"] = round(
            sum(d.n_boundary for d in sub_diagnostics) / len(sub_diagnostics), 2)
        results["sub_masked_avg_n_obstacle"] = round(
            sum(d.n_obstacle for d in sub_diagnostics) / len(sub_diagnostics), 2)
        results["sub_masked_avg_n_visited"]  = round(
            sum(d.n_visited  for d in sub_diagnostics) / len(sub_diagnostics), 2)
    else:
        results["sub_masked_avg_n_boundary"] = None
        results["sub_masked_avg_n_obstacle"] = None
        results["sub_masked_avg_n_visited"]  = None

    jsonl_results = dict(results)
    jsonl_results["termination_reason_raw"] = termination_reason_raw
    jsonl_results["route_types"]            = route_types
    jsonl_path = os.path.join(results_dir, "log", "evaluateroutes_log.jsonl")
    _append_jsonl(jsonl_path, jsonl_results)

    # Console output
    unsuccessful_routes = total_routes - successful
    denom = max(unsuccessful_routes, 1)

    def _pct(count):
        return (count * 100.0 / denom) if unsuccessful_routes > 0 else 0.0

    lines = [
        "",
        (f"Evaluate Routes — Used Dataset: {dataset_name} | final evaluation (checkpoint: "
         f"{config.get('inference_model_name', 'n/a')})"
         if str(dataset_name).lower() == "test" else
         f"Evaluate Routes — Used Dataset: {dataset_name} | Epoch {epoch}/{total_epochs}"),
        "--------------------------------------------------------------",
        f"--> Total Routes                 : {total_routes}/{total_samples}",
        f"--> Successful                   : {successful} ({success_rate:0.2f}%)",
        f"--> UnSuccessful                 : {total_routes - successful} ({((total_routes - successful) * 100.0 / max(total_routes, 1)):0.2f}%)",
        f"--> Step Efficiency              : {mean_step_ratio:0.3f}",
        f"--> Mean Extra Steps             : {mean_extra_steps:0.3f}",
        f"--> Exact Match Steps            : {exact_match_steps}",
        f"--> Fewer Steps Than Truth       : {fewer_steps_than_truth}",
        f"--> Reached Horizon (succ+fail)  : {hit_horizon_count} ({hit_horizon_rate:0.2f}%)",
        f"--> Horizon Successes            : {hit_horizon_success_count} ({horizon_success_rate:0.2f}%)",
        f"--> Horizon Failures (max_length): {horizon_fail_count}",
    ]

    # The five route types
    lines.append("--> Route types               |Color      |Count  |% routes|% failures")
    lines.append("==============================================================")
    for name in route_type_names:
        count = route_type_counts.get(name, 0)
        pct_all = count * 100.0 / max(total_routes, 1)
        pct_fail = "" if name == ROUTE_TYPE_SUCCESS else f"{_pct(count):7.2f}%"
        lines.append(f">>- {name:24s}:|{route_type_colors.get(name, 'n/a'):11s}|{count:7d}|{pct_all:7.2f}%|{pct_fail}")
    if integrity_not_masked or integrity_free_left:
        lines.append(f"--> [WARN] {integrity_not_masked} failure(s) not ended by masking, "
                     f"{integrity_free_left} with a free side left — check the rollout.")

    print("\n".join(lines))

    os.makedirs(results_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, "log", "evaluateroutes_log.csv")
    csv_results = dict(results)
    # Strip the per-sample JSON strings from the CSV row (too large / not useful there)
    csv_results.pop("termination_reason_raw", None)
    csv_results.pop("route_types",            None)
    _write_results_row(csv_path, csv_results)

    print(f"--> Evaluate Routes Logged to: {csv_path}")
    print("--------------------------------------------------------------")
    return results


def plot_model_overview_performance(stage=None):
    results_dir = config.get("results_dir", "results")
    base_dir    = os.path.join(results_dir, "model_performance")
    csv_path    = os.path.join(results_dir, "log", "evaluateroutes_log.csv")
    if not os.path.isfile(csv_path):
        print("No CSV log found — cannot plot overview.")
        return
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        print("No epoch column — cannot plot overview.")
        return

    metrics = [
        "success_rate", "mean_step_ratio", "mean_extra_steps",
        "exact_match_steps", "fewer_steps_than_truth", "hit_horizon_rate",
    ]
    metric_labels = {
        "success_rate":          "Success Rate (%)",
        "mean_step_ratio":       "Mean Step Ratio",
        "mean_extra_steps":      "Mean Extra Steps",
        "exact_match_steps":     "Exact Match Steps",
        "fewer_steps_than_truth":"Fewer Steps Than Truth",
        "hit_horizon_rate":      "Hit Horizon Rate (%)",
    }
    colors   = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    datasets = ["train", "val", "test"]
    if stage in datasets:
        datasets = [stage]

    for metric in metrics:
        metric_dir = os.path.join(base_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        legend_used = False

        for dataset in datasets:
            subset = df[df["dataset"] == dataset]
            if subset.empty or metric not in subset.columns:
                continue
            plt.plot(
                subset["epoch"], subset[metric],
                label=f"{dataset.capitalize()} {metric_labels.get(metric, metric)}",
                color=colors.get(dataset, "tab:gray"),
                marker="o", linewidth=2, markersize=8
            )
            legend_used = True

        plt.title(
            f"Model {metric_labels.get(metric, metric)} Over Epochs"
            + (f" — {stage.capitalize()}" if stage else "")
        )
        plt.xlabel("Epoch")
        plt.ylabel(metric_labels.get(metric, metric))
        plt.grid(True, linestyle="--", alpha=0.7)
        if legend_used:
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                       ncol=2, fontsize=12, frameon=False)
        plt.tight_layout()
        save_path = os.path.join(metric_dir, f"{metric}_performance_{stage or 'all'}.png")
        plt.savefig(save_path)
        plt.close()

def plot_loss_vs_success_rate(stage=None):
    results_dir = config.get("results_dir", "results")
    csv_path    = os.path.join(results_dir, "log", "evaluateroutes_log.csv")
    if not os.path.isfile(csv_path):
        print("No CSV log found — cannot plot loss vs success.")
        return

    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        print("CSV is empty or missing epoch — cannot plot loss vs success.")
        return

    datasets = ["train", "val", "test"]
    if stage in datasets:
        datasets = [stage]

    save_dir = os.path.join(results_dir, "model_performance")
    os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            continue
        if "loss_total" not in subset.columns or "success_rate" not in subset.columns:
            continue
        subset = subset.sort_values("epoch")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (loss_total)", color="tab:blue")
        ax1.plot(subset["epoch"], subset["loss_total"],
                 color="tab:blue", marker="o", linewidth=2, markersize=6, label="loss_total")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, linestyle="--", alpha=0.7)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Success Rate (%)", color="tab:orange")
        ax2.plot(subset["epoch"], subset["success_rate"],
                 color="tab:orange", marker="x", linewidth=2, markersize=6, label="success_rate")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title(f"Loss vs Success Rate ({dataset})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"loss_vs_success_{dataset}.png"))
        plt.close(fig)

def evaluate_after_epoch(
    predicted_routes,
    true_routes,
    goal_positions,
    dataset_path,
    epoch,
    total_samples=None,
    sample_indices=None,
    start_cells=None,
    run_id=None,
    contexts=None,
    termination_reasons=None,
    loss_total=None,
    loss_ce=None,
    loss_goal_delta=None,
    loss_goal_prox=None,
    loss_goal_obstacle=None,
):
    dataset_name = detect_dataset_name(dataset_path)
    return evaluate_routes(
        predicted_routes, true_routes, goal_positions, epoch, dataset_name,
        total_samples=total_samples, sample_indices=sample_indices,
        start_cells=start_cells, run_id=run_id, contexts=contexts,
        termination_reasons=termination_reasons,
        loss_total=loss_total, loss_ce=loss_ce,
        loss_goal_delta=loss_goal_delta, loss_goal_prox=loss_goal_prox,
        loss_goal_obstacle=loss_goal_obstacle,
    )

def plot_train_val_loss_and_val_success_panel():
    results_dir = config.get("results_dir", "results")
    csv_path    = os.path.join(results_dir, "log", "evaluateroutes_log.csv")
    if not os.path.isfile(csv_path):
        print(f"No CSV log found at {csv_path} — cannot plot panel.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"CSV log is empty at {csv_path} — cannot plot panel.")
        return

    if "loss_source" in df.columns:
        df = df[df["loss_source"].fillna("").isin(
            {"train_loop_epoch_avg", "route_eval_epoch_avg"}
        )]

    required_cols = {"dataset", "epoch", "loss_total", "success_rate", "loss_source"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"CSV missing required columns: {sorted(missing)} — cannot plot panel.")
        return

    train_df = df[(df["dataset"] == "train") &
                  (df["loss_source"] == "train_loop_epoch_avg")].sort_values("epoch")
    val_df   = df[(df["dataset"] == "val") &
                  (df["loss_source"] == "route_eval_epoch_avg")].sort_values("epoch")

    if train_df.empty and val_df.empty:
        print("No train(loop)/val(eval) rows found in CSV — cannot plot panel.")
        return

    base_dir = os.path.join(results_dir, "model_performance")
    os.makedirs(base_dir, exist_ok=True)

    fig, (ax_loss, ax_sr) = plt.subplots(1, 2, figsize=(12, 4.5))

    if not train_df.empty:
        ax_loss.plot(train_df["epoch"], train_df["loss_total"],
                     label="Train(Loop) Loss", marker="o", linewidth=2)
    if not val_df.empty:
        ax_loss.plot(val_df["epoch"], val_df["loss_total"],
                     label="Val(Eval) Loss", marker="o", linewidth=2)

    ax_loss.set_title("Train(Loop) Loss vs Val(Eval) Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (loss_total)")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend()

    if not val_df.empty:
        val_sr = val_df.copy()
        val_sr["success_rate"] = pd.to_numeric(val_sr["success_rate"], errors="coerce")
        val_sr = val_sr.dropna(subset=["success_rate"])
        if not val_sr.empty:
            ax_sr.plot(val_sr["epoch"], val_sr["success_rate"],
                       label="Val(Eval) Success Rate", color="tab:green",
                       marker="o", linewidth=2)

    ax_sr.set_title("Val(Eval) Success Rate")
    ax_sr.set_xlabel("Epoch")
    ax_sr.set_ylabel("Success Rate (%)")
    ax_sr.grid(True, linestyle="--", alpha=0.6)
    ax_sr.set_ylim(0, 100)
    ax_sr.legend()

    fig.tight_layout()
    out_path = os.path.join(base_dir, "train_val_loss_and_val_success.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved panel: {out_path}")

def log_epoch_losses_only(
    *,
    run_id,
    epoch,
    dataset_name,
    loss_total,
    loss_ce=None,
    loss_goal_delta=None,
    loss_goal_prox=None,
    loss_goal_obstacle=None,
    loss_source="train_loop_epoch_avg",
):

    dim_size     = config.get("dim")
    num_layers   = config.get("num_layers")
    num_heads    = config.get("heads")
    batch_size   = config.get("batch")
    total_epochs = config.get("epochs")
    results_dir  = config.get("results_dir", "results")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path  = os.path.join(results_dir, "log", "evaluateroutes_log.csv")

    results = {
        "run_id":      run_id,
        "timestamp":   timestamp,
        "dataset":     dataset_name,
        "epoch":       epoch,
        "dim_size":    dim_size,
        "num_layers":  num_layers,
        "num_heads":   num_heads,
        "batch_size":  batch_size,
        "total_epochs":total_epochs,

        "total_routes":       0,
        "successful_routes":  0,
        "unsuccessful_routes":0,
        "success_rate":       "",

        "mean_step_ratio":        "",
        "mean_extra_steps":       "",
        "exact_match_steps":      "",
        "fewer_steps_than_truth": "",

        "hit_horizon_count":         "",
        "hit_horizon_rate":          "",
        "hit_horizon_success_count": "",
        "hit_horizon_success_rate":  "",
        "hit_horizon_fail_count":    "",

        "loss_source":        loss_source,
        "loss_total":         _to_float_or_none(loss_total),
        "loss_ce":            _to_float_or_none(loss_ce),
        "loss_goal_delta":    _to_float_or_none(loss_goal_delta),
        "loss_goal_prox":     _to_float_or_none(loss_goal_prox),
        "loss_goal_obstacle": _to_float_or_none(loss_goal_obstacle),

        "ce_temperature":       _to_float_or_none(config.get("ce_temperature")),
        "goal_distance_weight": _to_float_or_none(config.get("goal_distance_weight")),
        "goal_proximity_weight":_to_float_or_none(config.get("goal_proximity_weight")),
        "goal_obstacle_weight": _to_float_or_none(config.get("goal_obstacle_weight")),

        **{f"route_{name}": "" for name in get_route_type_names()},
        "integrity_fail_not_masked": "",
        "integrity_fail_free_left":  "",
        "termination_counts":        "",
    }

    _write_results_row(csv_path, results)
    _append_jsonl(os.path.join(results_dir, "log", "evaluateroutes_log.jsonl"), results)
    return results
