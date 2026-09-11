"""PSTAR wrapper that maps latent outputs to U/D/R/L logits and computes the training loss."""
import torch
import torch.nn as nn
from config import config
from torch.nn import functional as F
from utils import mask_logits


class direction_model(nn.Module):
    """
    Wrapper for the transformer model to predict directions (Up, Down, Left, Right)
    with optional goal-directed loss computed on predicted positions.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # PerceiverARTransformer Block
        self._debug_batches_printed = 0
        self.last_losses = None

    def reset_debug_epoch(self):
        self._debug_batches_printed = 0
        self.last_losses = None

    @staticmethod
    def _build_obstacle_grid_from_context(input_tensor, *, context_len, n_rows, n_cols, device, dtype):
        """
        Build a [B, 1, H, W] grid with 1.0 for obstacle cells, else 0.0.
        Uses context tokens: [cell_num, x, y, State_Status, ...]
        """
        obstacle_status = int(config["status_obstacle"])

        status = input_tensor[:, :context_len, 3].detach().to(torch.int64)  # [B, C]
        x = input_tensor[:, :context_len, 1].detach().to(dtype=dtype)       # [B, C]
        y = input_tensor[:, :context_len, 2].detach().to(dtype=dtype)       # [B, C]

        is_obstacle = status.eq(obstacle_status)

        xi = (x - 1).round().to(torch.int64).clamp(0, n_cols - 1)
        yi = (y - 1).round().to(torch.int64).clamp(0, n_rows - 1)

        obs_grid = torch.zeros((input_tensor.size(0), 1, n_rows, n_cols), device=device, dtype=dtype)

        if is_obstacle.any():
            b_idx  = torch.arange(input_tensor.size(0), device=device).unsqueeze(1).expand_as(xi)
            b_flat = b_idx[is_obstacle]
            x_flat = xi[is_obstacle]
            y_flat = yi[is_obstacle]
            obs_grid.index_put_(
                (b_flat, torch.zeros_like(b_flat), y_flat, x_flat),
                torch.ones_like(x_flat, dtype=dtype),
                accumulate=False
            )

        return obs_grid

    @staticmethod
    def _sample_grid_bilinear(grid, pos_xy, *, n_rows, n_cols):
        """
        grid:    [B, 1, H, W]
        pos_xy:  [B, T, 2]  x,y are 1-indexed continuous coords in [1..W],[1..H]
        Returns: sampled values [B, T]
        """
        x = pos_xy[..., 0]
        y = pos_xy[..., 1]

        x_norm = torch.zeros_like(x) if n_cols <= 1 else (x - 1.0) / float(n_cols - 1) * 2.0 - 1.0
        y_norm = torch.zeros_like(y) if n_rows <= 1 else (y - 1.0) / float(n_rows - 1) * 2.0 - 1.0

        sample_grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # [B, T, 1, 2]

        sampled = F.grid_sample(
            grid, sample_grid,
            mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [B, 1, T, 1]

        return sampled[:, 0, :, 0]  # [B, T]

    def forward(self, input_tensor, masking_tensor, target_tensor):
        padding_value    = config["padding_value"]
        context_len      = int(config["context_matrix_length"])
        foundroute_len   = int(config["foundroute_matrix_length"])

        goal_distance_weight  = float(config.get("goal_distance_weight",  0.0))
        goal_proximity_weight = float(config.get("goal_proximity_weight", 0.0))
        obstacle_weight       = float(config.get("goal_obstacle_weight",  0.0))

        temperature = max(float(config.get("ce_temperature", 1.0)), 1e-6)

        debug_every_n_epochs       = int(config.get("debug_print_losses_every_n_epochs", 0))
        debug_max_prints_per_epoch = int(config.get("debug_print_losses_max_per_epoch", 1))

        batch_size = input_tensor.size(0)
        device     = input_tensor.device
        dtype      = torch.float32

        # ── Feature toggle flags (read once) ──
        use_dxdy = bool(config.get("feature_dx_dy",           False))
        use_f6   = bool(config.get("feature_free_neighbours",  False))
        use_f7   = bool(config.get("feature_is_interior",      False))

        # Feature index map from config (None when feature is disabled)
        fidx_dx         = config.get("fidx_dx")          # 4 or None
        fidx_dy         = config.get("fidx_dy")          # 5 or None
        fidx_free_nbrs  = config.get("fidx_free_nbrs")   # 6, 4, etc., or None
        fidx_interior   = config.get("fidx_is_interior") # 7, 5, etc., or None

        # ── Targets ──
        target_directions = torch.argmax(target_tensor, dim=-1)
        pad_row_sum = target_tensor.size(-1) * padding_value
        is_padded   = (target_tensor.sum(dim=-1) == pad_row_sum)
        target_directions = target_directions.masked_fill(is_padded, padding_value)
        target_directions_reorg = target_directions.reshape(-1).long()

        # ── Primary forward pass xinput uses all 175 tokens minus the last one (174 tokens) so that
        # to_logits(ff_tensor[:, -75:]) is aligned with latent positions 0..74.
        xinput             = input_tensor[:, :-1]
        transformer_output = self.base_model(xinput)    # [B, T, 4]

        masked_logits = mask_logits(
            logits=transformer_output, current_cells=None, masking_data=masking_tensor
        )  # [B, T, 4]

        model_output_reorg = masked_logits.reshape(-1, masked_logits.size(-1))
        if temperature != 1.0:
            model_output_reorg = model_output_reorg / temperature

        # ── Cross-entropy loss ──
        ce_loss = F.cross_entropy(
            model_output_reorg,
            target_directions_reorg,
            ignore_index=padding_value
        )

        # ── Goal-directed & obstacle losses ──
        # Guard behind weight check — avoids expensive rollout when all weights=0.
        distance_loss = torch.zeros((), device=device, dtype=dtype)
        proximity_loss = torch.zeros((), device=device, dtype=dtype)
        obstacle_loss  = torch.zeros((), device=device, dtype=dtype)

        any_goal_loss = (goal_distance_weight > 0.0 or
                         goal_proximity_weight > 0.0 or
                         obstacle_weight > 0.0)

        if any_goal_loss:
            # Goal position
            end_status_val = int(config["status_end"])
            context_status = input_tensor[:, :context_len, 3].to(torch.int64)
            goal_mask  = context_status.eq(end_status_val)
            has_goal   = goal_mask.any(dim=1)
            goal_idx   = goal_mask.to(torch.int64).argmax(dim=1)
            batch_idx  = torch.arange(batch_size, device=device)
            goal_xy    = input_tensor[batch_idx, goal_idx, 1:3].to(dtype=dtype)
            start_xy   = input_tensor[:, context_len, 1:3].to(dtype=dtype)
            goal_xy    = torch.where(has_goal.unsqueeze(1), goal_xy, start_xy)

            probs  = torch.softmax(masked_logits, dim=-1).to(dtype=dtype)  # [B, T, 4]
            n_rows = int(config["grid_size"][0])
            n_cols = int(config["grid_size"][1])

            # Straight-through position rollout
            positions = [start_xy]
            for step in range(foundroute_len - 1):
                cur        = positions[-1]
                step_probs = probs[:, step]
                hard_dir   = torch.argmax(masked_logits[:, step], dim=-1)
                one_hot    = F.one_hot(hard_dir, 4).to(dtype=dtype)
                one_hot_st = one_hot - step_probs.detach() + step_probs
                step_dx    = one_hot_st[:, 2] - one_hot_st[:, 3]
                step_dy    = one_hot_st[:, 1] - one_hot_st[:, 0]
                nxt = torch.stack([cur[:, 0] + step_dx, cur[:, 1] + step_dy], dim=-1)
                nxt = torch.stack(
                    [torch.clamp(nxt[:, 0], 1.0, float(n_cols)),
                     torch.clamp(nxt[:, 1], 1.0, float(n_rows))], dim=-1
                )
                step_is_padded = is_padded[:, step]
                nxt = torch.where(step_is_padded.unsqueeze(1), cur, nxt)
                positions.append(nxt)

            predicted_positions = torch.stack(positions, dim=1)  # [B, T, 2]
            goal_expanded = goal_xy.unsqueeze(1)
            distances = torch.abs(predicted_positions - goal_expanded).sum(dim=-1)  # [B, T]

            if goal_distance_weight > 0.0 and distances.size(1) > 1:
                distance_deltas = distances[:, 1:] - distances[:, :-1]
                valid_mask = ~is_padded[:, :distances.size(1) - 1]
                if valid_mask.any():
                    distance_increase = torch.relu(distance_deltas)
                    weighted_penalty  = (distance_increase * valid_mask.to(dtype)).sum()
                    num_valid = valid_mask.sum().clamp(min=1).to(dtype)
                    distance_loss = weighted_penalty / num_valid

            if goal_proximity_weight > 0.0:
                valid_steps_full = ~is_padded[:, :distances.size(1)]
                if valid_steps_full.any():
                    proximity_loss = distances[valid_steps_full].mean()

            if obstacle_weight > 0.0:
                obs_grid = self._build_obstacle_grid_from_context(
                    input_tensor,
                    context_len=context_len, n_rows=n_rows, n_cols=n_cols,
                    device=device, dtype=dtype
                )
                obstacle_values = self._sample_grid_bilinear(
                    obs_grid, predicted_positions, n_rows=n_rows, n_cols=n_cols
                )
                valid_steps_full = ~is_padded[:, :obstacle_values.size(1)]
                if valid_steps_full.any():
                    obstacle_loss = obstacle_values[valid_steps_full].mean()

        total_loss = (
            ce_loss
            + goal_distance_weight  * distance_loss
            + goal_proximity_weight * proximity_loss
            + obstacle_weight       * obstacle_loss
        )

        self.last_losses = {
            "loss_total":           float(total_loss.detach().item()),
            "loss_ce":              float(ce_loss.detach().item()),
            "loss_goal_delta":      float(distance_loss.detach().item()),
            "loss_goal_prox":       float(proximity_loss.detach().item()),
            "loss_goal_obstacle":   float(obstacle_loss.detach().item()),
            "goal_distance_weight":  float(goal_distance_weight),
            "goal_proximity_weight": float(goal_proximity_weight),
            "goal_obstacle_weight":  float(obstacle_weight),
            "ce_temperature":        float(temperature),
        }

        # ── Optional debug print ──
        current_epoch = config.get("current_epoch")
        is_last_batch = bool(config.get("is_last_batch", False))

        do_epoch_gate = (
            debug_every_n_epochs > 0 and
            isinstance(current_epoch, int) and
            ((current_epoch + 1) % debug_every_n_epochs == 0)
        )
        should_print = (
            do_epoch_gate and
            is_last_batch and
            self._debug_batches_printed < debug_max_prints_per_epoch
        )

        if should_print:
            stage = str(config.get("debug_stage_label", "")).strip().lower()
            if not stage:
                stage = "Training" if self.training else "Eval"
            print(
                f"[{stage}] "
                f"Loss_Total={float(total_loss.item()):.6f} | "
                f"Loss_CE={float(ce_loss.item()):.6f} | "
                f"Loss_Goal_Delta={float(distance_loss.item()):.6f} (w={goal_distance_weight:g}) | "
                f"Loss_Goal_Prox.={float(proximity_loss.item()):.6f} (w={goal_proximity_weight:g}) | "
                f"Loss_Obstacle={float(obstacle_loss.item()):.6f} (w={obstacle_weight:g}) | "
                f"Epoch={current_epoch + 1}"
            )
            self._debug_batches_printed += 1

        return total_loss

    # =========================================================================
    # get_context_embeddings
    # =========================================================================
    def get_context_embeddings(self, input_tensor: torch.Tensor) -> dict:
        """
        Extract context token embeddings.

        """
        # ── Step 1: Read config values ────────────────────────────────────────
        context_length = int(config["context_matrix_length"])   # 100
        use_mean      = bool(config.get("ragate_use_mean_emb",        True))
        use_start     = bool(config.get("ragate_use_start_emb",       False))
        use_goal      = bool(config.get("ragate_use_goal_emb",        False))
        all_layers    = bool(config.get("ragate_embedding_all_layers", False))

        num_layers = len(self.base_model.layers)   # e.g. 12

        # ── Step 2: Register hooks ────────────────────────────────────────────
        # all_layers=False: hook last FeedForward only → 1 captured tensor
        # all_layers=True:  hook all FeedForward blocks → num_layers tensors
        #                   captured list is ordered layer 0 → layer N-1
        captured  = []
        handles   = []

        def make_hook():
            def hook_fn(module, input, output):
                captured.append(output.detach())
            return hook_fn

        if all_layers:
            for i, (attn_prenorm, ff_prenorm) in enumerate(self.base_model.layers):
                target = ff_prenorm.fn
                assert type(target).__name__ == "FeedForward", (
                    f"Layer {i} hook target is not FeedForward: "
                    f"{type(target).__name__}. Check transformer.py."
                )
                handles.append(target.register_forward_hook(make_hook()))
        else:
            target = self.base_model.layers[-1][1].fn
            assert type(target).__name__ == "FeedForward", (
                f"Hook target is not FeedForward: {type(target).__name__}. "
                f"Check transformer.py layer structure."
            )
            handles.append(target.register_forward_hook(make_hook()))

        # ── Step 3: Forward pass under no_grad; always remove hooks ──────────
        try:
            if config.get("ragate_pstar_frozen", True):
                with torch.no_grad():
                    _ = self.base_model(input_tensor)
            else:
                _ = self.base_model(input_tensor)
        finally:
            for h in handles:
                h.remove()

        # ── Step 4: Build context embeddings ──────────────────────────────────
        # captured[i] shape: [B, sequence_len, dim]
        # Context slice:     [:, :context_length, :] = [:, :100, :]
        #
        # all_layers=False: captured has 1 tensor → context_mean [B, dim]
        # all_layers=True:  captured has num_layers tensors
        #                   → context_mean [B, dim * num_layers]  (cat along dim=1)

        result = {}

        if use_mean:
            if all_layers:
                # Each layer's context mean: [B, dim]
                # Stack then cat: [B, dim * num_layers]
                layer_means = []
                for layer_output in captured:          # ordered layer 0..N-1
                    ctx = layer_output[:, :context_length, :]   # [B, 100, dim]
                    layer_means.append(ctx.mean(dim=1))          # [B, dim]
                result["context_mean"] = torch.cat(layer_means, dim=1)  # [B, dim*N]
            else:
                ctx = captured[0][:, :context_length, :]         # [B, 100, dim]
                result["context_mean"] = ctx.mean(dim=1)          # [B, dim]

        # ── Step 5: Start and goal embeddings (always from last layer) ────────
        # start_emb and goal_emb use the last captured tensor regardless of
        # all_layers mode — last layer has the most task-relevant signal.
        if use_start or use_goal:
            last_output  = captured[-1]                           # [B, S, dim]
            context_embs = last_output[:, :context_length, :]    # [B, 100, dim]
            status_channel = int(config.get("fidx_status", 3))
            status_col = input_tensor[:, :context_length, status_channel]  # [B, 100]
            batch_idx  = torch.arange(input_tensor.shape[0],
                                      device=input_tensor.device)

            if use_start:
                start_status = int(config["status_start"])
                start_idx    = (status_col == start_status).float().argmax(dim=1)
                result["start_emb"] = context_embs[batch_idx, start_idx, :]  # [B, dim]

            if use_goal:
                goal_status = int(config["status_end"])
                goal_idx    = (status_col == goal_status).float().argmax(dim=1)
                result["goal_emb"] = context_embs[batch_idx, goal_idx, :]    # [B, dim]

        return result


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))

    from config import config
    from transformer import PerceiverARTransformer

    device = torch.device("cpu")

    # Build a minimal model (no checkpoint needed for shape tests)
    base_model = PerceiverARTransformer(
        num_unique_tokens = config["num_unique_tokens"],
        dim               = config["dim"],
        num_layers        = config["num_layers"],
        heads             = config["heads"],
        sequence_len      = config["sequence_len"],
        causal            = config["causal"],
        ff_mult           = config["ff_mult"],
        ff_dropout        = 0.0,
        attn_dropout      = 0.0,
    ).to(device)
    model = direction_model(base_model).to(device)
    model.eval()

    # Dummy input: [2, 175, feature_size]
    fs = config["feature_size"]
    seq_len = config["sequence_len"]
    dummy = torch.zeros(2, seq_len, fs, device=device)
    # Give batch item 0 a start (status=1) and goal (status=2) in context
    dummy[0, 0, 3] = 1.0   # context index 0 → start
    dummy[0, 1, 3] = 2.0   # context index 1 → goal

    # ── Test A: mean only (default) ──────────────────────────────────────────
    config["ragate_use_mean_emb"]  = True
    config["ragate_use_start_emb"] = False
    config["ragate_use_goal_emb"]  = False
    result = model.get_context_embeddings(dummy)
    assert set(result.keys()) == {"context_mean"}, f"Test A keys wrong: {result.keys()}"
    assert result["context_mean"].shape == (2, config["dim"]), \
        f"Test A shape wrong: {result['context_mean'].shape}"
    assert not result["context_mean"].requires_grad, "Test A: grad should be False"

    # ── Test B: goal only (no start, no mean) ────────────────────────────────
    config["ragate_use_mean_emb"]  = False
    config["ragate_use_start_emb"] = False
    config["ragate_use_goal_emb"]  = True
    result = model.get_context_embeddings(dummy)
    assert set(result.keys()) == {"goal_emb"}, f"Test B keys wrong: {result.keys()}"
    assert result["goal_emb"].shape == (2, config["dim"]), \
        f"Test B shape wrong: {result['goal_emb'].shape}"

    # ── Test C: all three active ──────────────────────────────────────────────
    config["ragate_use_mean_emb"]  = True
    config["ragate_use_start_emb"] = True
    config["ragate_use_goal_emb"]  = True
    result = model.get_context_embeddings(dummy)
    assert set(result.keys()) == {"context_mean", "start_emb", "goal_emb"}, \
        f"Test C keys wrong: {result.keys()}"
    assert result["start_emb"].shape == (2, config["dim"]), \
        f"Test C start_emb shape wrong: {result['start_emb'].shape}"
    assert result["goal_emb"].shape  == (2, config["dim"]), \
        f"Test C goal_emb shape wrong: {result['goal_emb'].shape}"

    # Restore defaults
    config["ragate_use_mean_emb"]  = True
    config["ragate_use_start_emb"] = False
    config["ragate_use_goal_emb"]  = False

    print("get_context_embeddings OK — all toggle combinations verified")
