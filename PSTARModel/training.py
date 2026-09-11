"""PSTAR training loop (autoregressive teacher forcing) + validation rollout."""
import os
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "1200000"

import time
import torch
import numpy as np
from config import config
from utils import (mask_logits, build_dynamic_mask,
                   combine_action_masks, get_next_cell_info,
                   directions_to_cell_path, decode_routes_with_masks,
                   _sync_if_cuda)
from dataloader import get_dataloader, PathfindingDataset
from route_evaluation import evaluate_after_epoch, log_epoch_losses_only
from route_evaluation import plot_train_val_loss_and_val_success_panel
from transformer import PerceiverARTransformer
from direction_model import direction_model
from visualization import visualize_training_samples, visualize_validation_samples
from torch.amp import autocast, GradScaler
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def _fmt_duration(seconds):
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.2f}m"
    return f"{minutes / 60.0:.2f}h"


def _get_lr(optimizer):
    """Return current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]

def _accumulate_losses(loss_sums, last_losses):
    if not last_losses:
        return
    for k in ("loss_total", "loss_ce", "loss_goal_delta", "loss_goal_prox", "loss_goal_obstacle"):
        v = last_losses.get(k)
        if v is None:
            continue
        loss_sums[k] = loss_sums.get(k, 0.0) + float(v)


def _finalize_losses(loss_sums, batch_count):
    if batch_count <= 0:
        return {}
    return {
        "loss_total":         loss_sums.get("loss_total", 0.0) / batch_count,
        "loss_ce":            loss_sums.get("loss_ce", 0.0) / batch_count,
        "loss_goal_delta":    loss_sums.get("loss_goal_delta", 0.0) / batch_count,
        "loss_goal_prox":     loss_sums.get("loss_goal_prox", 0.0) / batch_count,
        "loss_goal_obstacle": loss_sums.get("loss_goal_obstacle", 0.0) / batch_count,
    }


def train_epoch(model, dataloader, optimizer, epoch, device, dataset_label):
    model.train()
    config["debug_stage_label"] = "train"
    model.reset_debug_epoch()
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    running_loss = 0.0
    batch_count  = 0
    loss_sums    = {}

    total_batches = len(dataloader)
    epoch_start   = time.perf_counter()

    # REMOVED: tourcost_tensor unpacked from dataloader but never used anywhere.
    for batch_idx, (input_tensor, masking_tensor, target_tensor, contexts, found_routes_list) in enumerate(dataloader):
        batch_start = time.perf_counter()

        input_tensor   = input_tensor.to(device)    # [B, sequence_len, feature_size]
        masking_tensor = masking_tensor.to(device)   # [B, foundroute_len, num_directions]
        target_tensor  = target_tensor.to(device)    # [B, target_len, num_unique_tokens]

        # ── Sanity check: print first batch of first epoch only ──
        if batch_idx == 0 and epoch == 0:
            _ctx_len = config["context_matrix_length"]
            _use_dxdy = bool(config.get("feature_dx_dy",           False))
            _use_f6   = bool(config.get("feature_free_neighbours",  False))
            _use_f7   = bool(config.get("feature_is_interior",      False))
            _fidx_dx  = config.get("fidx_dx")
            _fidx_dy  = config.get("fidx_dy")
            _fidx_f6  = config.get("fidx_free_nbrs")
            _fidx_f7  = config.get("fidx_is_interior")

            # Build a human-readable expected-feature label from active toggles
            _feat_labels = ["cell_num", "x", "y", "state_status"]
            if _use_dxdy:
                _feat_labels += ["dx_to_goal", "dy_to_goal"]
            if _use_f6:
                _feat_labels.append("free_neighbours")
            if _use_f7:
                _feat_labels.append("is_interior")
            _expected_str = ", ".join(_feat_labels)

            print("\n" + "=" * 80)
            print("SANITY CHECK: First Batch, First Sample Features")
            print("=" * 80)
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Feature size from config: {config['feature_size']}")

            print(f"\nFirst CONTEXT cell (index 0):")
            print(f"  Features: {input_tensor[0, 0, :].cpu().numpy()}")
            print(f"  Expected: [{_expected_str}]")

            print(f"\nFirst PATH step (index {_ctx_len}):")
            print(f"  Features: {input_tensor[0, _ctx_len, :].cpu().numpy()}")
            print(f"  Expected: [{_expected_str}]")

            # dx/dy stats — only when the feature is active
            if _use_dxdy and _fidx_dx is not None and _fidx_dy is not None:
                _dx_ctx  = input_tensor[0, :_ctx_len, _fidx_dx]
                _dy_ctx  = input_tensor[0, :_ctx_len, _fidx_dy]
                _dx_path = input_tensor[0, _ctx_len:,  _fidx_dx]
                _dy_path = input_tensor[0, _ctx_len:,  _fidx_dy]
                print(f"\nGoal feature statistics (dx_to_goal, dy_to_goal):")
                print(f"  Context - Mean dx: {_dx_ctx.mean():.2f}, Mean dy: {_dy_ctx.mean():.2f}")
                print(f"  Path    - Mean dx: {_dx_path.mean():.2f}, Mean dy: {_dy_path.mean():.2f}")

            # free_neighbours stats — only when f6 active
            if _use_f6 and _fidx_f6 is not None:
                _f6_ctx = input_tensor[0, :_ctx_len, _fidx_f6]
                print(f"\nFree-neighbour feature (f6) — Context mean: {_f6_ctx.mean():.3f}")

            # is_interior stats — only when f7 active
            if _use_f7 and _fidx_f7 is not None:
                _f7_ctx = input_tensor[0, :_ctx_len, _fidx_f7]
                print(f"Is-interior feature    (f7) — Context mean: {_f7_ctx.mean():.3f}")

            # Goal cell features
            for i in range(_ctx_len):
                if input_tensor[0, i, 3] == config["status_end"]:
                    print(f"\nGoal cell found at context index {i}:")
                    print(f"  Features: {input_tensor[0, i, :].cpu().numpy()}")
                    if _use_dxdy:
                        print(f"  (dx=0, dy=0 expected for goal cell itself)")
                    break

            print("=" * 80 + "\n")

        # ── Training mode: autoregressive

        context_len   = int(config["context_matrix_length"])
        latent_len    = input_tensor.shape[1] - context_len
        padding_value = config["padding_value"]
        training_mode = str(config.get("training_mode", "autoregressive")).strip().lower()

        # ── Latent tail debug (first batch of first epoch only)
        if batch_idx == 0 and epoch == 0:
            # context_len already defined above in this function
            latent_tail  = input_tensor[0, context_len + 1:, :]
            print("\n" + "=" * 80)
            print("MIN DEBUG: Latent tail check (sample 0)")
            print(f"latent_tail.shape: {tuple(latent_tail.shape)}")
            print(f"latent_tail.sum(): {float(latent_tail.sum().item()):.6f}")
            print(f"latent_tail.abs().sum(): {float(latent_tail.abs().sum().item()):.6f}")
            print("latent_tail first row:", latent_tail[0].detach().cpu().numpy() if latent_tail.numel() else "N/A")
            print("=" * 80 + "\n")

        config["current_epoch"] = int(epoch)
        config["is_last_batch"] = (batch_idx + 1 == total_batches)
        config["run_id"] = RUN_ID

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            loss = model(input_tensor, masking_tensor, target_tensor)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_value"])
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        batch_count  += 1
        _accumulate_losses(loss_sums, getattr(model, "last_losses", None))

        # ── Training visualisation (last batch of epoch only) ──
        if config.get("enable_training_visualization", False) and (batch_idx + 1 == total_batches):
            with torch.no_grad():
                pred_cell_paths, _ = decode_routes_with_masks(
                    model=model,
                    input_tensor=input_tensor,
                    masking_tensor=masking_tensor,
                    contexts=contexts,
                    grid_size=config["grid_size"],
                    max_steps=int(config["foundroute_matrix_length"]),
                    blank_latent_tail=True
                )
            visualize_training_samples(
                model=model, epoch=epoch, config=config,
                input_tensor=input_tensor, masking_tensor=masking_tensor,
                target_tensor=target_tensor, contexts=contexts,
                batch_idx=batch_idx, found_routes_list=found_routes_list,
                pred_routes=pred_cell_paths
            )

    _sync_if_cuda(device)
    avg_loss   = running_loss / batch_count if batch_count > 0 else 0.0
    avg_losses = _finalize_losses(loss_sums, batch_count)

    return avg_loss, avg_losses


def train_model():
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
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
    trainable_params     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Total parameters: {trainable_params + non_trainable_params} "
          f"(Trainable = {trainable_params}, Non-Trainable = {non_trainable_params})")

    if config.get("optimizer", "adam").lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["learning_rate"], momentum=config.get("momentum", 0.0)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], **config.get("adam_params", {})
        )

    # Resume from checkpoint (optional)
    resume_path = config.get("resume_from_checkpoint", "")
    start_best_val_loss         = float("inf")
    start_best_val_success_rate = float("-inf")
    if resume_path and os.path.exists(resume_path):
        print(f"[RESUME] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[k] = val.to(device)
        start_best_val_loss         = float(ckpt.get("validating_loss", float("inf")))
        start_best_val_success_rate = float(ckpt.get("val_success_rate", float("-inf")))
        loaded_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"[RESUME] Loaded from epoch {loaded_epoch} | "
              f"val_loss={start_best_val_loss:.6f} | "
              f"val_success={start_best_val_success_rate:.2f}%")
    elif resume_path:
        print(f"[RESUME] WARNING: checkpoint not found: {resume_path} — training from scratch")

    # ── torch.compile (optional, PyTorch 2.0+) 
    if config.get("use_torch_compile", False) and hasattr(torch, "compile"):
        print("[COMPILE] Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("[COMPILE] Done — first batch will trigger JIT compilation (~60-90s)")
    elif config.get("use_torch_compile", False):
        print("[COMPILE] torch.compile requested but not available "
              "(requires PyTorch >= 2.0) — skipping")

    train_dataset = PathfindingDataset(config["train_file"], config, "train_file_pickle")
    val_dataset   = PathfindingDataset(config["val_file"],   config, "val_file_pickle")

    val_loader = get_dataloader(config["val_file"], config, "val_file_pickle", base_dataset=val_dataset)

    if val_loader is None:
        print("Error: Could not create data loaders.")
        return

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_val_loss         = start_best_val_loss
    best_val_success_rate = start_best_val_success_rate

    total_epochs = config["epochs"]

    for epoch in range(total_epochs):
        train_loader = get_dataloader(
            config["train_file"],
            config,
            "train_file_pickle",
            base_dataset=train_dataset,
        )

        train_epoch_start = time.perf_counter()
        train_loss, train_avg_losses = train_epoch(
            model, train_loader, optimizer, epoch, device, dataset_label="train dataset"
        )
        _sync_if_cuda(device)
        train_epoch_time = time.perf_counter() - train_epoch_start

        print(
            f"Training(train dataset) - Epoch Loss {epoch+1}/{total_epochs} = {train_loss:.6f} "
            f"(CE={train_avg_losses.get('loss_ce', 0.0):.6f}, "
            f"Goal_Delta={train_avg_losses.get('loss_goal_delta', 0.0):.6f}, "
            f"Goal_Prox={train_avg_losses.get('loss_goal_prox', 0.0):.6f}, "
            f"Obstacle={train_avg_losses.get('loss_goal_obstacle', 0.0):.6f})"
            f" - Time = {_fmt_duration(train_epoch_time)}"
        )

        log_epoch_losses_only(
            run_id=RUN_ID, epoch=epoch + 1, dataset_name="train",
            loss_total=train_avg_losses.get("loss_total", train_loss),
            loss_ce=train_avg_losses.get("loss_ce"),
            loss_goal_delta=train_avg_losses.get("loss_goal_delta"),
            loss_goal_prox=train_avg_losses.get("loss_goal_prox"),
            loss_goal_obstacle=train_avg_losses.get("loss_goal_obstacle"),
            loss_source="train_loop_epoch_avg",
        )

        (avg_val_loss, avg_val_acc, val_pred_routes, val_true_routes,
         val_goal_positions, val_sample_indices, val_start_cells,
         val_contexts, val_termination_reasons, val_last_losses,
         val_failed_indices) = validate(
            model, val_loader, epoch, device, dataset_label="val dataset"
        )

        val_results = evaluate_after_epoch(
            predicted_routes=val_pred_routes, true_routes=val_true_routes,
            goal_positions=val_goal_positions, sample_indices=val_sample_indices,
            start_cells=val_start_cells, dataset_path=config["val_file"],
            epoch=epoch + 1, total_samples=len(val_loader.dataset),
            run_id=RUN_ID, contexts=val_contexts,
            termination_reasons=val_termination_reasons,
            loss_total=val_last_losses.get("loss_total"),
            loss_ce=val_last_losses.get("loss_ce"),
            loss_goal_delta=val_last_losses.get("loss_goal_delta"),
            loss_goal_prox=val_last_losses.get("loss_goal_prox"),
            loss_goal_obstacle=val_last_losses.get("loss_goal_obstacle"),
        )

        ckpt_base = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validating_loss': avg_val_loss,
        }
        torch.save(ckpt_base, os.path.join(config["checkpoint_dir"], config["last_model_name"]))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_base, os.path.join(config["checkpoint_dir"], config["best_model_name"]))

        val_success_rate = val_results.get("success_rate") if isinstance(val_results, dict) else None
        if val_success_rate is not None and val_success_rate > best_val_success_rate:
            best_val_success_rate = val_success_rate
            torch.save(
                {**ckpt_base, 'val_success_rate': val_success_rate},
                os.path.join(config["checkpoint_dir"], config["ce_best_model_name"])
            )

    if config.get("plot_model_overview_performance", True):
        plot_train_val_loss_and_val_success_panel()

    print("=== Training Completed ===")


def validate(model, dataloader, epoch, device, dataset_label=""):
    model.eval()

    stage_label = "train_eval" if "train" in str(dataset_label).lower() else "val"
    config["debug_stage_label"] = stage_label
    model.reset_debug_epoch()

    running_loss    = 0.0
    running_correct = 0
    running_total   = 0
    batch_count     = 0
    loss_sums       = {}

    context_len   = config["context_matrix_length"]
    foundroute_len = config["foundroute_matrix_length"]
    padding_value  = config["padding_value"]
    total_batches  = len(dataloader)
    epoch_start    = time.perf_counter()

    all_pred_routes        = []
    all_true_routes        = []
    all_goal_positions     = []
    all_sample_indices     = []
    all_start_cells        = []
    all_contexts           = []
    all_termination_reasons = []
    failed_sample_indices = []

    with torch.no_grad():
        for batch_idx, (input_tensor, masking_tensor, target_tensor, contexts, found_routes_list) in enumerate(dataloader):
            input_tensor   = input_tensor.to(device)
            masking_tensor = masking_tensor.to(device)
            target_tensor  = target_tensor.to(device)

            config["current_epoch"] = int(epoch)
            config["is_last_batch"] = (batch_idx + 1 == total_batches)
            config["run_id"] = RUN_ID

            batch_size        = input_tensor.shape[0]
            batch_start_index = batch_idx * dataloader.batch_size

            # ── Pass 1: loss ──
            loss = model(input_tensor, masking_tensor, target_tensor)
            running_loss += float(loss.item())
            batch_count  += 1
            _accumulate_losses(loss_sums, getattr(model, "last_losses", None))

            # ── Pass 2: autoregressive rollout via shared utility ──
            pred_cell_paths, termination_reasons = decode_routes_with_masks(
                model=model,
                input_tensor=input_tensor,
                masking_tensor=masking_tensor,
                contexts=contexts,
                grid_size=config["grid_size"],
                max_steps=foundroute_len,
                blank_latent_tail=True
            )

            state_status_maps = [
                {cell.get("Cell_Number", 0): cell.get("State_Status", 0) for cell in contexts[b]}
                for b in range(batch_size)
            ]

            _fidx_st   = int(config["fidx_status"])
            _fidx_cell = int(config["fidx_cell_num"])
            goal_cells = []
            for b in range(batch_size):
                goal_cell = None
                for i in range(context_len):
                    if int(input_tensor[b, i, _fidx_st].item()) == config["status_end"]:
                        goal_cell = int(input_tensor[b, i, _fidx_cell].item())
                        break
                if goal_cell is None:
                    goal_cell = int(input_tensor[b, context_len, _fidx_cell].item())
                goal_cells.append(goal_cell)

            # ── Validation visualisation (last batch of epoch only) ──
            if config.get("enable_validation_visualization", False) and (batch_idx + 1 == total_batches):
                visualize_validation_samples(
                    model=model, epoch=epoch, config=config,
                    input_tensor=input_tensor, masking_tensor=masking_tensor,
                    target_tensor=target_tensor, contexts=contexts,
                    batch_idx=batch_idx,
                    pred_routes=pred_cell_paths,
                    found_routes_list=found_routes_list
                )

            for b in range(batch_size):
                start_cell    = int(input_tensor[b, context_len, _fidx_cell].item())
                dataset_index = batch_start_index + b
                all_sample_indices.append(dataset_index)
                all_start_cells.append(start_cell)
                all_contexts.append(contexts[b])
                all_goal_positions.append(goal_cells[b])
                all_termination_reasons.append(
                    termination_reasons[b] if termination_reasons else None
                )
                all_pred_routes.append(pred_cell_paths[b])

                pred_last = pred_cell_paths[b][-1] if pred_cell_paths[b] else None
                if pred_last is None or int(pred_last) != int(goal_cells[b]):
                    failed_sample_indices.append(dataset_index)

            for b in range(batch_size):
                found_routes = found_routes_list[b]
                true_route = [
                    cell for cell in found_routes
                    if cell.get("Cell_Number", config["padding_value"]) != config["padding_value"]
                ]
                all_true_routes.append(true_route)

    _sync_if_cuda(device)
    epoch_time = time.perf_counter() - epoch_start

    avg_loss   = running_loss / batch_count if batch_count > 0 else 0.0
    avg_losses = _finalize_losses(loss_sums, batch_count)

    print(
        f"Validation({dataset_label}) - Epoch Loss {epoch+1}/{config['epochs']} = {avg_loss:.6f} "
        f"(CE={avg_losses.get('loss_ce', 0.0):.6f}, "
        f"Goal_Delta={avg_losses.get('loss_goal_delta', 0.0):.6f}, "
        f"Goal_Prox={avg_losses.get('loss_goal_prox', 0.0):.6f}, "
        f"Obstacle={avg_losses.get('loss_goal_obstacle', 0.0):.6f})"
        f" - Time = {_fmt_duration(epoch_time)}"
    )

    return (avg_loss, 0.0, all_pred_routes, all_true_routes, all_goal_positions,
            all_sample_indices, all_start_cells, all_contexts, all_termination_reasons,
            avg_losses, failed_sample_indices)


if __name__ == "__main__":
    train_model()
