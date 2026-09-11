"""PSTAR route visualization (true vs predicted, per-subtype failure plots)."""
import os
import matplotlib
import torch
import numpy as np
from config import config
from unittest.mock import patch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataloader import get_dataloader
from utils import directions_to_cell_path, get_next_cell_info
from route_evaluation import classify_failure_reason
from stuck_masked_subtypes import classify_stuck_masked


def _viz_outcome_folder(pred_cells, goal_cell, context, config):
    """Visualization subfolder for a route: 'reached_goal' if it ends at the goal,
    else the stuck_masked subtype (isolated/boundary_corner/boundary_side/self_trap)."""
    if not pred_cells or goal_cell is None:
        return "reached_goal"
    if int(pred_cells[-1]) == int(goal_cell):
        return "reached_goal"
    ssmap = {c.get("Cell_Number", 0): c.get("State_Status", 0) for c in context}
    ma = classify_stuck_masked(
        last_cell=int(pred_cells[-1]), pred_cells=pred_cells, goal_cell=int(goal_cell),
        grid_size=config["grid_size"], state_status_map=ssmap,
        visited=set(int(c) for c in pred_cells), config=config,
    )
    return ma.sub_reason
import pandas as pd


def get_state_status_list(context, grid_size):
    n_cells = grid_size[0] * grid_size[1]
    state_status_list = [0] * n_cells
    for cell in context:
        cell_num = cell.get("Cell_Number", 0)
        status = cell.get("State_Status", 0)
        idx = cell_num - 1  # for 1-indexed
        if 0 <= idx < n_cells:
            state_status_list[idx] = status
    return state_status_list


def get_obstacle_coords(state_status_list, grid_size, status_obstacle):
    obstacles = set()
    for idx, status in enumerate(state_status_list):
        if status == status_obstacle:
            x = idx % grid_size[1] + 1
            y = idx // grid_size[1] + 1
            obstacles.add((x, y))
    return obstacles


def _is_int_like(value):
    return isinstance(value, (int, np.integer))


def _sanitize_directions(seq):
    if not seq:
        return []
    sanitized = []
    for x in seq:
        if not _is_int_like(x):
            continue
        xi = int(x)
        if 0 <= xi <= 3:
            sanitized.append(xi)
    return sanitized


def _sanitize_cell_path(seq, grid_size):
    if not seq:
        return []
    max_cell = int(grid_size[0] * grid_size[1])
    sanitized = []
    for x in seq:
        if not _is_int_like(x):
            continue
        xi = int(x)
        if 1 <= xi <= max_cell:
            sanitized.append(xi)
    return sanitized


def _to_cell_path(route, start_cell_num, grid_size, state_status_map, route_kind="auto"):
    """
    route_kind:
        - "cells": route is already a list of cell numbers
        - "directions": route is a list of directions (0..3)
        - "auto": best-effort detection (not recommended for stable visuals)
    """
    if route is None:
        return [start_cell_num]

    if route_kind == "cells":
        cells = _sanitize_cell_path(route, grid_size)
        return cells if cells else [start_cell_num]

    if route_kind == "directions":
        dirs = _sanitize_directions(route)
        if not dirs:
            return [start_cell_num]
        return directions_to_cell_path(start_cell_num, dirs, grid_size, state_status_map)

    # auto: prefer directions only if every element is a valid direction
    dirs = _sanitize_directions(route)
    if dirs and len(dirs) == len(route):
        return directions_to_cell_path(start_cell_num, dirs, grid_size, state_status_map)

    cells = _sanitize_cell_path(route, grid_size)
    return cells if cells else [start_cell_num]


def _truncate_at_goal(cells, goal_cell):
    if not cells or goal_cell is None:
        return cells

    for i, c in enumerate(cells):
        if c == goal_cell:
            return cells[:i + 1]
    return cells


def plot_routes(
    true_route,
    pred_route,
    grid_size,
    start_cell_num,
    obstacles=None,
    save_path=None,
    show=config["show_visualizations"],
    title=None,
    context=None,
    intended_goal_cells=None,
    pred_title="Predicted Route",
    true_route_kind="auto",
    pred_route_kind="auto",
    pred_failure_reason=None
):
    state_status_list = get_state_status_list(context, grid_size)
    state_status_map = {i + 1: status for i, status in enumerate(state_status_list)}

    # Build cell -> (x, y, dx, dy) maps from context for labeling
    cell_xy = {}
    cell_dxdy = {}

    goal_x = None
    goal_y = None
    if context:
        for cell in context:
            if int(cell.get("State_Status", 0)) == int(config["status_end"]):
                goal_x = cell.get("x")
                goal_y = cell.get("y")
                break

        for cell in context:
            cell_num = cell.get("Cell_Number")
            x = cell.get("x")
            y = cell.get("y")
            if cell_num is None or x is None or y is None:
                continue

            try:
                cell_num_i = int(cell_num)
            except (TypeError, ValueError):
                continue

            cell_xy[cell_num_i] = (x, y)

            # Prefer stored dx/dy if present; otherwise compute from goal
            dx = cell.get("dx_to_goal", None)
            dy = cell.get("dy_to_goal", None)

            if dx is None or dy is None:
                if goal_x is not None and goal_y is not None:
                    try:
                        dx = float(goal_x) - float(x)
                        dy = float(goal_y) - float(y)
                    except (TypeError, ValueError):
                        dx = None
                        dy = None

            cell_dxdy[cell_num_i] = (dx, dy)

    true_cells = _to_cell_path(true_route, start_cell_num, grid_size, state_status_map, route_kind=true_route_kind)
    pred_cells = _to_cell_path(pred_route, start_cell_num, grid_size, state_status_map, route_kind=pred_route_kind)

    if pred_title == "Teacher-Forced Route":
        pred_cells = _truncate_at_goal(pred_cells, intended_goal_cells)

    failure_color_map = {
        "stuck_full_obstacle":   "purple",
        "stuck_boundary_corner": "deepskyblue",
        "stuck_revisit_3plus":   "gold",
        "stuck_all_masked":      "orange",
        "max_steps_unreached":   "black",
        "failed_wrong_direction":"magenta",
        "failed_loops":          "cyan",
        "failed_empty_path":     "gray",
        "unknown_stuck":         "brown",
    }

    failure_color_for_legend = failure_color_map.get(pred_failure_reason, "red")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ["True Route", pred_title]
    routes = [true_cells, pred_cells]
    route_colors = ["g", "r"]
    route_styles = ["--o", "--x"]

    show_dxdy = bool(config.get("viz_show_dxdy_in_cells", False))

    for idx, ax in enumerate(axes):
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                cell_num = row * grid_size[1] + col + 1

                label = str(cell_num)
                if show_dxdy:
                    dx, dy = cell_dxdy.get(cell_num, (None, None))
                    if dx is not None and dy is not None:
                        # format as integers when possible, else one decimal
                        try:
                            dx_f = float(dx)
                            dy_f = float(dy)
                            dx_s = f"{int(dx_f)}" if dx_f.is_integer() else f"{dx_f:.1f}"
                            dy_s = f"{int(dy_f)}" if dy_f.is_integer() else f"{dy_f:.1f}"
                            label = f"{cell_num}\ndx={dx_s}\ndy={dy_s}"
                        except (TypeError, ValueError):
                            label = str(cell_num)

                ax.text(col + 1, row + 1, label, va="center", ha="center", fontsize=7, color="gray")
                rect = plt.Rectangle((col + 0.5, row + 0.5), 1, 1, fill=False, edgecolor="lightgray", linewidth=0.5)
                ax.add_patch(rect)

        ax.axhline(-0.5, color="black", linewidth=1.5)
        ax.axvline(-0.5, color="black", linewidth=1.5)

        if obstacles:
            for x, y in obstacles:
                ax.fill(
                    [x - 0.5, x + 0.5, x + 0.5, x - 0.5],
                    [y - 0.5, y - 0.5, y + 0.5, y + 0.5],
                    "k",
                    alpha=0.5
                )

        route = routes[idx]
        color = route_colors[idx]
        style = route_styles[idx]
        collision_cells = set()

        sx = (start_cell_num - 1) % grid_size[1] + 1
        sy = (start_cell_num - 1) // grid_size[1] + 1

        if len(route) > 1:
            x_route = [(cell - 1) % grid_size[1] + 1 for cell in route if cell > 0]
            y_route = [(cell - 1) // grid_size[1] + 1 for cell in route if cell > 0]
            ax.plot(x_route, y_route, style, color=color, label=titles[idx])

            start_rect = plt.Rectangle(
                (sx - 0.5, sy - 0.5),
                1,
                1,
                facecolor="lime",
                edgecolor="none",
                alpha=0.5
            )
            ax.add_patch(start_rect)

            end_x = x_route[-1]
            end_y = y_route[-1]

            reached_goal = (intended_goal_cells is not None and route[-1] == intended_goal_cells)
            if reached_goal:
                ax.plot(end_x, end_y, "*", color="red", markersize=16, markeredgecolor="k", label="Goal")
            else:
                failure_color = failure_color_map.get(pred_failure_reason, "red")
                failure_label = f"Failed ({pred_failure_reason})" if pred_failure_reason else "Failed"
                ax.plot(end_x, end_y, "*", color=failure_color, markersize=16, markeredgecolor="k", label=failure_label)

            if obstacles:
                for x, y in zip(x_route, y_route):
                    if (x, y) in obstacles:
                        ax.plot(x, y, "mx", markersize=12, label="Collision" if not collision_cells else "")
                        collision_cells.add((x, y))
        else:
            start_rect = plt.Rectangle(
                (sx - 0.5, sy - 0.5),
                1,
                1,
                facecolor="lime",
                edgecolor="none",
                alpha=0.5
            )
            ax.add_patch(start_rect)

            failure_color = failure_color_map.get(pred_failure_reason, "red")
            failure_label = f"Failed ({pred_failure_reason})" if pred_failure_reason else "Failed"
            ax.plot(sx, sy, "*", color=failure_color, markersize=16, markeredgecolor="k", label=failure_label)

        ax.set_title(titles[idx])
        ax.set_xlim(0.5, grid_size[1] + 0.5)
        ax.set_ylim(0.5, grid_size[0] + 0.5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")

        if idx == 0:
            legend_elements = [
                mpatches.Patch(color="lime", label="Start", alpha=0.5),
                Line2D([0], [0], marker="*", color="w", label="Goal (reached)", markerfacecolor="red", markeredgecolor="k", markersize=12),
                Line2D([0], [0], color="g", marker="o", linestyle="--", label="True Route", markersize=6, linewidth=1),
                Line2D([0], [0], marker="x", color="m", label="Collision", markersize=6, linewidth=1),
                mpatches.Patch(color="k", label="Obstacle", alpha=0.5),
                mpatches.Patch(color="orange", label="Intended Goal", alpha=0.5),
            ]
        else:
            legend_elements = [
                mpatches.Patch(color="lime", label="Start", alpha=0.5),
                Line2D([0], [0], marker="*", color="w", label="Goal (reached)", markerfacecolor="red", markeredgecolor="k", markersize=12),
                Line2D([0], [0], color="r", marker="x", linestyle="--", label=pred_title, markersize=6, linewidth=1),
                Line2D([0], [0], marker="x", color="m", label="Collision", markersize=6, linewidth=1),
                mpatches.Patch(color="k", label="Obstacle", alpha=0.5),
                mpatches.Patch(color="orange", label="Intended Goal", alpha=0.5),
            ]

        if pred_failure_reason:
            legend_elements.append(Line2D([0], [0], marker="*", color="w", label=f"Failed ({pred_failure_reason})",
                                          markerfacecolor=failure_color_for_legend, markeredgecolor="k",
                                          markersize=12))

        ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

        if intended_goal_cells is not None:
            x_goal = (intended_goal_cells - 1) % grid_size[1] + 1
            y_goal = (intended_goal_cells - 1) // grid_size[1] + 1

            goal_rect = plt.Rectangle(
                (x_goal - 0.5, y_goal - 0.5),
                1,
                1,
                facecolor="orange",
                edgecolor="none",
                alpha=0.5
            )
            ax.add_patch(goal_rect)

    fig.subplots_adjust(wspace=0.30)

    if title:
        fig.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show and matplotlib.get_backend() != "Agg":
        plt.show()
    plt.close(fig)


def visualize_training_samples(model, epoch, config, input_tensor, masking_tensor, target_tensor, contexts, batch_idx=0, found_routes_list=None, pred_routes=None):
    if config.get("num_train_visualizations", 0) <= 0:
        return
    if not config.get("enable_training_visualization", False):
        return

    batch_size = input_tensor.shape[0]
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    masking_tensor = masking_tensor.to(device)
    target_tensor = target_tensor.to(device)

    pred_tf = None
    if pred_routes is None and config.get("show_predicted_route_in_training", False):
        with torch.no_grad():
            logits = model.base_model(input_tensor, masking_tensor)
            logits_latent = logits[:, -config["foundroute_matrix_length"]:, :]
            logits_skip_first = logits_latent[:, 1:, :]
            pred_tf = logits_skip_first.argmax(dim=2)

    max_vis = int(config.get("num_train_visualizations", 0))
    max_vis = max(0, min(max_vis, batch_size))
    sample_indices = list(range(max_vis))
    for vis_count, sample_idx in enumerate(sample_indices):
        true_route = []
        if found_routes_list is not None:
            true_route = [
                cell.get("Cell_Number")
                for cell in found_routes_list[sample_idx]
                if cell.get("Cell_Number", config["padding_value"]) != config["padding_value"]
            ]
        else:
            target_vecs = target_tensor[sample_idx].cpu().numpy()
            true_indices = []
            for idx in target_vecs:
                if np.all(idx == config["padding_value"]):
                    break
                true_indices.append(int(idx) if np.isscalar(idx) else int(np.argmax(idx)))
            true_route = true_indices

        pred_route = None
        pred_title = "Teacher-Forced Route"
        pred_route_kind = "directions"

        if pred_routes is not None:
            pred_route = pred_routes[sample_idx]
            pred_title = "Predicted Route"

            # AUTO-DETECT: directions (0..3) vs cells (1..N)
            sanitized_dirs = _sanitize_directions(pred_route)
            has_only_dirs_or_pad = all((x == config["padding_value"]) or (0 <= int(x) <= 3) for x in pred_route if _is_int_like(x))
            pred_route_kind = "directions" if sanitized_dirs and has_only_dirs_or_pad else "cells"

        elif pred_tf is not None:
            pred_route = pred_tf[sample_idx].cpu().numpy().tolist()
            pred_route_kind = "directions"

        context = contexts[sample_idx]
        state_status_list = get_state_status_list(context, config["grid_size"])
        obstacles = get_obstacle_coords(state_status_list, config["grid_size"], config["status_obstacle"])
        start_cell_num = int(input_tensor[sample_idx, config["context_matrix_length"], 0].item())

        goal_cell = None
        for cell in context:
            if cell.get("State_Status") == config["status_end"]:
                goal_cell = int(cell.get("Cell_Number"))
                break

        pred_failure_reason = None
        pred_cells_for_reason = None
        if goal_cell is not None and pred_route is not None:
            state_status_map = {i + 1: status for i, status in enumerate(state_status_list)}
            pred_cells_for_reason = _to_cell_path(
                pred_route,
                start_cell_num,
                config["grid_size"],
                state_status_map,
                route_kind=pred_route_kind
            )
            pred_failure_reason = classify_failure_reason(
                pred_route=pred_cells_for_reason,
                true_route=true_route,
                goal_cell=goal_cell,
                context=context,
                grid_size=config["grid_size"],
                max_path_len=int(config["foundroute_matrix_length"]) + 1
            )
            if pred_failure_reason == "success":
                pred_failure_reason = None

        save_path = None
        if config.get("save_visualizations", True):
            vis_dir = os.path.join(config.get("results_dir", "results/"), "visualization")
            _folder = _viz_outcome_folder(pred_cells_for_reason, goal_cell, context, config)
            save_dir = os.path.join(vis_dir, _folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"train_epoch_{epoch+1}_batch_{batch_idx+1}_sample_{sample_idx+1}.png")

        try:
            plot_routes(
                true_route=true_route,
                pred_route=pred_route,
                grid_size=config["grid_size"],
                start_cell_num=start_cell_num,
                obstacles=obstacles,
                save_path=save_path,
                show=config.get("show_visualizations", False),
                title=f"Route Visualization During Training | Epoch {epoch+1} | Batch {batch_idx+1} | Sample {sample_idx+1}",
                context=context,
                intended_goal_cells=goal_cell,
                pred_title=pred_title,
                true_route_kind="cells",
                pred_route_kind=pred_route_kind,
                pred_failure_reason=pred_failure_reason
            )
        except Exception as e:
            print(f"Training visualization failed: {e}")


def visualize_validation_samples(model, epoch, config, input_tensor, masking_tensor, target_tensor, contexts, batch_idx=0, pred_routes=None, found_routes_list=None,
                                 termination_reasons=None):
    if config.get("num_val_visualizations", 0) <= 0:
        return
    if not config.get("enable_validation_visualization", False):
        return

    batch_size = input_tensor.shape[0]
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    masking_tensor = masking_tensor.to(device)
    target_tensor = target_tensor.to(device)

    pred_tf = None
    if pred_routes is None and config.get("show_predicted_route_in_training", False):
        with torch.no_grad():
            logits = model.base_model(input_tensor, masking_tensor)
            logits_latent = logits[:, -config["foundroute_matrix_length"]:, :]
            logits_skip_first = logits_latent[:, 1:, :]
            pred_tf = logits_skip_first.argmax(dim=2)

    max_vis = int(config.get("num_val_visualizations", 0))
    max_vis = max(0, min(max_vis, batch_size))
    sample_indices = list(range(max_vis))
    for vis_count, sample_idx in enumerate(sample_indices):
        true_route = []
        if found_routes_list is not None:
            true_route = [
                cell.get("Cell_Number")
                for cell in found_routes_list[sample_idx]
                if cell.get("Cell_Number", config["padding_value"]) != config["padding_value"]
            ]
        else:
            target_vecs = target_tensor[sample_idx].cpu().numpy()
            true_indices = []
            for idx in target_vecs:
                if np.all(idx == config["padding_value"]):
                    break
                true_indices.append(int(idx) if np.isscalar(idx) else int(np.argmax(idx)))
            true_route = true_indices

        pred_route = None
        pred_title = "Teacher-Forced Route"
        pred_route_kind = "directions"

        if pred_routes is not None:
            pred_route = pred_routes[sample_idx]
            pred_title = "Predicted Route"
            # pred_routes from decode_routes_with_masks are cell numbers (1-100),
            # NOT directions (0-3). Auto-detect so the plot renders the route correctly.
            if pred_route is not None:
                sanitized_dirs = _sanitize_directions(pred_route)
                if sanitized_dirs and len(sanitized_dirs) == len(pred_route):
                    pred_route_kind = "directions"
                else:
                    pred_route_kind = "cells"
        elif pred_tf is not None:
            pred_route = pred_tf[sample_idx].cpu().numpy().tolist()
            pred_route_kind = "directions"

        context = contexts[sample_idx]
        state_status_list = get_state_status_list(context, config["grid_size"])
        obstacles = get_obstacle_coords(state_status_list, config["grid_size"], config["status_obstacle"])
        start_cell_num = int(input_tensor[sample_idx, config["context_matrix_length"], 0].item())

        goal_cell = None
        for cell in context:
            if cell.get("State_Status") == config["status_end"]:
                goal_cell = int(cell.get("Cell_Number"))
                break

        pred_failure_reason = None
        pred_cells_for_reason = None
        reached_goal_val = False
        folder_reason = None

        if goal_cell is not None and pred_route is not None:
            state_status_map = {i + 1: status for i, status in enumerate(state_status_list)}
            pred_cells_for_reason = _to_cell_path(
                pred_route,
                start_cell_num,
                config["grid_size"],
                state_status_map,
                route_kind=pred_route_kind
            )
            term_reason = (
                termination_reasons[sample_idx]
                if termination_reasons and sample_idx < len(termination_reasons)
                else "unknown_stuck"
            )
            pred_failure_reason = classify_failure_reason(
                pred_route=pred_cells_for_reason,
                true_route=true_route,
                goal_cell=goal_cell,
                context=context,
                grid_size=config["grid_size"],
                max_path_len=int(config["foundroute_matrix_length"]) + 1,
                term_reason=term_reason
            )
            if pred_failure_reason in ("success", "reached_goal"):
                pred_failure_reason = None

            if pred_cells_for_reason:
                reached_goal_val = int(pred_cells_for_reason[-1]) == int(goal_cell)

            folder_reason = pred_failure_reason
            if pred_failure_reason == "stuck_all_masked" and pred_cells_for_reason:
                ssmap = {cell.get("Cell_Number", 0): cell.get("State_Status", 0) for cell in context}
                ma = classify_stuck_masked(
                    last_cell        = pred_cells_for_reason[-1],
                    pred_cells       = pred_cells_for_reason,
                    goal_cell        = goal_cell,
                    grid_size        = config["grid_size"],
                    state_status_map = ssmap,
                    visited          = set(pred_cells_for_reason),
                    config           = config,
                )
                folder_reason = ma.sub_reason

        save_path = None
        if config.get("save_visualizations", True):
            vis_dir = os.path.join(config.get("results_dir", "results/"), "visualization")
            fname = f"val_epoch_{epoch+1}_batch_{batch_idx+1}_sample_{sample_idx+1}.png"
            _folder = _viz_outcome_folder(pred_cells_for_reason, goal_cell, context, config)
            save_dir = os.path.join(vis_dir, _folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, fname)

        try:
            plot_routes(
                true_route=true_route,
                pred_route=pred_route,
                grid_size=config["grid_size"],
                start_cell_num=start_cell_num,
                obstacles=obstacles,
                save_path=save_path,
                show=config.get("show_visualizations", False),
                title=f"Route Visualization During Validation | Epoch {epoch+1} | Batch {batch_idx+1} | Sample {sample_idx+1}",
                context=context,
                intended_goal_cells=goal_cell,
                pred_title=pred_title,
                true_route_kind="cells",
                pred_route_kind=pred_route_kind,
                pred_failure_reason=pred_failure_reason
            )
        except Exception as e:
            print(f"Validation visualization failed: {e}")

def visualize_inference_samples(pred_routes, input_tensor, target_tensor, contexts, config, get_state_status_list, get_obstacle_coords, plot_routes,
                                goal_cells=None, epoch=0, batch_idx=0, goal_reached=None, found_routes_list=None, run_id=None,
                                termination_reasons=None):
    max_per_folder = int(config.get("num_inference_visualizations", 20))
    save_vis = config.get("save_visualizations", False)
    show_vis = config.get("show_visualizations", False)
    vis_dir = os.path.join(config.get("results_dir", "results/"), "visualization")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if batch_idx == 0 or not hasattr(visualize_inference_samples, "_folder_counts"):
        visualize_inference_samples._folder_counts = {}
    folder_counts = visualize_inference_samples._folder_counts
    batch_size = input_tensor.shape[0]
    for sample_idx in range(batch_size):
        pred_route = pred_routes[sample_idx]
        context = contexts[sample_idx]
        state_status_list = get_state_status_list(context, config["grid_size"])
        obstacles = get_obstacle_coords(state_status_list, config["grid_size"], config["status_obstacle"])
        start_cell_num = int(input_tensor[sample_idx, config["context_matrix_length"], 0].item())
        intended_goal_cell = goal_cells[sample_idx] if goal_cells is not None else None

        true_route = []
        if found_routes_list is not None:
            true_route = [
                cell.get("Cell_Number")
                for cell in found_routes_list[sample_idx]
                if cell.get("Cell_Number", config["padding_value"]) != config["padding_value"]
            ]

        # AUTO-DETECT: directions (0..3) vs cells (1..N)
        pred_route_kind = "auto"
        if pred_route is not None:
            sanitized_dirs = _sanitize_directions(pred_route)
            if sanitized_dirs and len(sanitized_dirs) == len(pred_route):
                pred_route_kind = "directions"
            else:
                pred_route_kind = "cells"

        pred_failure_reason = None
        pred_cells_for_reason = None
        reached_goal = False
        folder_reason = None

        if intended_goal_cell is not None and pred_route is not None:
            state_status_map = {i + 1: status for i, status in enumerate(state_status_list)}
            pred_cells_for_reason = _to_cell_path(
                pred_route,
                start_cell_num,
                config["grid_size"],
                state_status_map,
                route_kind=pred_route_kind
            )
            term_reason = (
                termination_reasons[sample_idx]
                if termination_reasons and sample_idx < len(termination_reasons)
                else "unknown_stuck"
            )
            pred_failure_reason = classify_failure_reason(
                pred_route=pred_cells_for_reason,
                true_route=true_route,
                goal_cell=int(intended_goal_cell),
                context=context,
                grid_size=config["grid_size"],
                max_path_len=int(config["foundroute_matrix_length"]) + 1,
                term_reason=term_reason
            )
            if pred_failure_reason in ("reached_goal", "success"):
                pred_failure_reason = None

            if pred_cells_for_reason:
                reached_goal = int(pred_cells_for_reason[-1]) == int(intended_goal_cell)

            folder_reason = pred_failure_reason
            if pred_failure_reason == "stuck_all_masked" and pred_cells_for_reason:
                ssmap = {cell.get("Cell_Number", 0): cell.get("State_Status", 0) for cell in context}
                ma = classify_stuck_masked(
                    last_cell        = pred_cells_for_reason[-1],
                    pred_cells       = pred_cells_for_reason,
                    goal_cell        = int(intended_goal_cell),
                    grid_size        = config["grid_size"],
                    state_status_map = ssmap,
                    visited          = set(pred_cells_for_reason),
                    config           = config,
                )
                folder_reason = ma.sub_reason

        save_path = None
        _folder = None
        if save_vis:
            _folder = _viz_outcome_folder(pred_cells_for_reason, intended_goal_cell, context, config)
            if folder_counts.get(_folder, 0) >= max_per_folder:
                continue   # enough examples already saved for this outcome folder
            run_tag = run_id or "inference"
            fname = f"{run_tag}_batch_{batch_idx+1}_sample_{sample_idx+1}.png"
            save_dir = os.path.join(vis_dir, _folder)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, fname)

        try:
            plot_routes(
                true_route=true_route,
                pred_route=pred_route,
                grid_size=config["grid_size"],
                start_cell_num=start_cell_num,
                obstacles=obstacles,
                save_path=save_path,
                show=show_vis,
                title=f"Route Visualization During Inference | Epoch {epoch+1} | Batch {batch_idx+1} | Sample {sample_idx+1}",
                context=context,
                intended_goal_cells=intended_goal_cell,
                pred_title="Predicted Route",
                true_route_kind="cells",
                pred_route_kind=pred_route_kind,
                pred_failure_reason=pred_failure_reason
            )
        except Exception as e:
            print("Plotting failed:", e)

        if save_vis and _folder is not None:
            folder_counts[_folder] = folder_counts.get(_folder, 0) + 1

