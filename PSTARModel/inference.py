"""PSTAR inference: checkpoint loading and the autoregressive test-set rollout."""
import os
import torch
import matplotlib
matplotlib.use('Agg')
from config import config
from utils import (mask_logits, get_next_cell_info, build_dynamic_mask,
                   combine_action_masks, directions_to_cell_path,
                   _write_next_token, _sync_if_cuda)
from dataloader import get_dataloader, get_ragate_dataloader
from route_evaluation import evaluate_after_epoch
from transformer import PerceiverARTransformer
from direction_model import direction_model
from visualization import (plot_routes, get_state_status_list, get_obstacle_coords,
                            visualize_inference_samples)
# Import shared loss helpers from training instead of duplicating them here.
from training import _accumulate_losses, _finalize_losses, RUN_ID


def load_model_from_checkpoint(checkpoint_path, device):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model = PerceiverARTransformer(
        num_unique_tokens=config["num_unique_tokens"],
        dim=config["dim"],
        num_layers=config["num_layers"],
        heads=config["heads"],
        sequence_len=config["sequence_len"],
        causal=config["causal"],
        ff_mult=config["ff_mult"],
        ff_dropout=config["ff_dropout"],
        attn_dropout=config["attn_dropout"]
    ).to(device)
    model = direction_model(base_model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
        # else: stays 0.0 from blanked tail


def inference_main():
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(
        config["checkpoint_dir"],
        config.get("inference_model_name", config["ce_best_model_name"])
    )
    model = load_model_from_checkpoint(checkpoint_path, device)
    test_loader = get_dataloader(config["test_file"], config, "test_file_pickle")
    if test_loader is None:
        print(f"ERROR: Could not load test data from {config['test_file']}")
        return

    model.eval()
    context_len    = config["context_matrix_length"]
    foundroute_len = config["foundroute_matrix_length"]
    padding_value  = config["padding_value"]

    # Feature index constants — read from config, no hardcoding
    fidx_cell = int(config["fidx_cell_num"])
    fidx_x    = int(config["fidx_x"])
    fidx_y    = int(config["fidx_y"])
    fidx_st   = int(config["fidx_status"])

    config["debug_stage_label"] = "test"
    model.reset_debug_epoch()

    all_pred_routes         = []
    all_true_routes         = []
    all_goal_positions      = []
    all_sample_indices      = []
    all_start_cells         = []
    all_contexts            = []
    all_termination_reasons = []

    loss_sums        = {}
    loss_batch_count = 0

    with torch.no_grad():
        for batch_idx, (input_tensor, masking_tensor, target_tensor, contexts, found_routes_list) in enumerate(test_loader):
            input_tensor   = input_tensor.to(device)
            masking_tensor = masking_tensor.to(device)
            target_tensor  = target_tensor.to(device)

            # Loss pass (for reporting only — no backward).
            _ = model(input_tensor, masking_tensor, target_tensor)
            _accumulate_losses(loss_sums, getattr(model, "last_losses", None))
            loss_batch_count += 1

            batch_size        = input_tensor.shape[0]
            batch_start_index = batch_idx * test_loader.batch_size
            current_input     = input_tensor.clone()
            predictions       = []

            state_status_maps = [
                {cell.get("Cell_Number", 0): cell.get("State_Status", 0) for cell in contexts[b]}
                for b in range(batch_size)
            ]

            # ── Find goal cells ──
            goal_cells    = []
            goal_cells_xy = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
            for b in range(batch_size):
                goal_cell = goal_x = goal_y = None
                for i in range(context_len):
                    if int(current_input[b, i, fidx_st].item()) == int(config["status_end"]):
                        goal_cell = int(current_input[b, i, fidx_cell].item())
                        goal_x    = float(current_input[b, i, fidx_x].item())
                        goal_y    = float(current_input[b, i, fidx_y].item())
                        break
                if goal_cell is None:
                    goal_cell = int(current_input[b, context_len, fidx_cell].item())
                    goal_x    = float(current_input[b, context_len, fidx_x].item())
                    goal_y    = float(current_input[b, context_len, fidx_y].item())
                goal_cells.append(goal_cell)
                goal_cells_xy[b, 0] = goal_x
                goal_cells_xy[b, 1] = goal_y

            visited_cells = [set() for _ in range(batch_size)]
            for b in range(batch_size):
                visited_cells[b].add(int(current_input[b, context_len, fidx_cell].item()))

            done               = [False] * batch_size
            termination_reason = [None]  * batch_size
            hit_horizon        = [True]  * batch_size
            goal_reached       = [False] * batch_size

            # Blank the latent tail before autoregressive rollout.
            latent_len = current_input.shape[1] - context_len
            if latent_len > 1:
                current_input[:, context_len + 1:, :] = float(config["padding_value"])

            # ── Autoregressive rollout ──
            for step in range(foundroute_len):
                dynamic_masks = []
                any_active    = False

                for b in range(batch_size):
                    current_cell = int(current_input[b, context_len + step, 0].item())

                    if done[b]:
                        mask_vec = [0] * config["num_unique_tokens"]
                    else:
                        any_active = True
                        prev_cell  = (int(current_input[b, context_len + step - 1, 0].item())
                                      if step > 0 else None)

                        mask_vec = build_dynamic_mask(
                            current_cell, config["grid_size"],
                            state_status_maps[b], visited_cells[b]
                        )

                        if bool(config.get("rollout_use_dataset_mask", False)):
                            mask_vec = combine_action_masks(
                                dynamic_mask_vec=mask_vec,
                                dataset_mask_vec=masking_tensor[b, step].tolist(),
                                padding_value=padding_value,
                                num_actions=config["num_unique_tokens"]
                            )

                        if mask_vec == [0] * config["num_unique_tokens"]:
                            done[b]               = True
                            termination_reason[b] = "stuck_masked"
                            hit_horizon[b]        = False

                    dynamic_masks.append(mask_vec)

                mask_step            = torch.tensor(dynamic_masks, dtype=torch.float32,
                                                    device=current_input.device).unsqueeze(1)
                logits               = model.base_model(current_input)
                logits_step          = logits[:, step:step + 1, :]
                current_cells_tensor = current_input[:, context_len + step, 0].unsqueeze(1)
                masked_logits        = mask_logits(logits_step, current_cells_tensor, mask_step)

                pred = masked_logits.argmax(dim=-1)
                for b in range(batch_size):
                    if done[b]:
                        pred[b, 0] = padding_value
                predictions.append(pred)

                for b in range(batch_size):
                    if done[b]:
                        continue
                    direction    = int(pred[b, 0].item())
                    current_cell = int(current_cells_tensor[b, 0].item())
                    next_info    = get_next_cell_info(
                        current_cell, direction, config["grid_size"], state_status_maps[b]
                    )
                    if next_info is None:
                        done[b]               = True
                        termination_reason[b] = "stuck_boundary"
                        hit_horizon[b]        = False
                        continue

                    next_cell, x, y, state_status = next_info
                    visited_cells[b].add(int(next_cell))

                    if int(next_cell) == int(goal_cells[b]):
                        done[b]               = True
                        goal_reached[b]       = True
                        termination_reason[b] = "success"
                        hit_horizon[b]        = False

                    if step < foundroute_len - 1:
                        _write_next_token(
                            current_input, b,
                            slot=context_len + step + 1,
                            next_cell=next_cell, x=x, y=y,
                            state_status=state_status,
                            goal_cells_xy=goal_cells_xy,
                            context_len=context_len,
                        )

                if any_active and all(done):
                    for b in range(batch_size):
                        if termination_reason[b] is None:
                            hit_horizon[b] = False
                    break

            for b in range(batch_size):
                if termination_reason[b] is None:
                    termination_reason[b] = "max_length" if hit_horizon[b] else "unknown_stuck"

            pred_seq_np = torch.cat(predictions, dim=1).cpu().numpy().tolist()

            for b in range(batch_size):
                all_sample_indices.append(batch_start_index + b)
                all_start_cells.append(int(input_tensor[b, context_len, 0].item()))
                all_contexts.append(contexts[b])
                all_goal_positions.append(goal_cells[b])
                all_termination_reasons.append(termination_reason[b])

                directions_pred = [d for d in pred_seq_np[b] if d != padding_value]
                all_pred_routes.append(
                    directions_to_cell_path(
                        all_start_cells[-1], directions_pred,
                        config["grid_size"], state_status_maps[b]
                    )
                )
                all_true_routes.append([
                    cell for cell in found_routes_list[b]
                    if cell.get("Cell_Number", padding_value) != padding_value
                ])

            if config.get("enable_inference_visualization", False):
                visualize_inference_samples(
                    pred_routes=all_pred_routes[-batch_size:],
                    input_tensor=input_tensor,
                    target_tensor=target_tensor,
                    contexts=contexts,
                    config=config,
                    get_state_status_list=get_state_status_list,
                    get_obstacle_coords=get_obstacle_coords,
                    plot_routes=plot_routes,
                    goal_cells=goal_cells,
                    epoch=0,
                    batch_idx=batch_idx,
                    goal_reached=goal_reached,
                    found_routes_list=found_routes_list,
                    run_id=RUN_ID,
                    termination_reasons=termination_reason,
                )

    _sync_if_cuda(device)
    test_avg_losses = _finalize_losses(loss_sums, loss_batch_count)

    evaluate_after_epoch(
        predicted_routes=all_pred_routes,
        true_routes=all_true_routes,
        goal_positions=all_goal_positions,
        sample_indices=all_sample_indices,
        start_cells=all_start_cells,
        dataset_path=config["test_file"],
        epoch=1,
        total_samples=len(test_loader.dataset),
        run_id=RUN_ID,
        contexts=all_contexts,
        termination_reasons=all_termination_reasons,
        loss_total=test_avg_losses.get("loss_total"),
        loss_ce=test_avg_losses.get("loss_ce"),
        loss_goal_delta=test_avg_losses.get("loss_goal_delta"),
        loss_goal_prox=test_avg_losses.get("loss_goal_prox"),
        loss_goal_obstacle=test_avg_losses.get("loss_goal_obstacle"),
    )

