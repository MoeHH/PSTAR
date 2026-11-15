import torch
from config import config
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=None, deterministic=None, verbose=True):
    import os as _os
    import random as _random
    import numpy as _np
    seed = int(config.get("random_seed", 42) if seed is None else seed)
    deterministic = bool(config.get("deterministic", True) if deterministic is None else deterministic)
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        if "CUBLAS_WORKSPACE_CONFIG" not in _os.environ:
            print("  [SEED] WARNING: CUBLAS_WORKSPACE_CONFIG is not set — GPU matmuls may "
                  "not be deterministic. Start the run through main.py.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
    if verbose:
        print(f"  [SEED] seed={seed}  deterministic={deterministic}")
    return seed

def make_generator(offset=0):
    g = torch.Generator()
    g.manual_seed(int(config.get("random_seed", 42)) + int(offset))
    return g

def seed_worker(worker_id):
    import random as _random
    import numpy as _np
    worker_seed = torch.initial_seed() % 2 ** 32
    _np.random.seed(worker_seed)
    _random.seed(worker_seed)

def rng_state():
    import random as _random
    import numpy as _np
    return {
        "python": _random.getstate(),
        "numpy":  _np.random.get_state(),
        "torch":  torch.get_rng_state(),
        "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def set_rng_state(state):
    import random as _random
    import numpy as _np
    if not state:
        return
    if state.get("python") is not None:
        _random.setstate(state["python"])
    if state.get("numpy") is not None:
        _np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def get_backbone_class(architecture=None):
    arch = str(config.get("architecture", "pstar") if architecture is None else architecture).lower()
    if arch == "pstar":
        from pstar_transformer import PSTARTransformer
        return PSTARTransformer
    if arch == "perceiverar":
        from perceiverar_transformer import PerceiverARTransformer
        return PerceiverARTransformer
    raise ValueError(f"config['architecture']={arch!r} — use 'pstar' or 'perceiverar'.")

def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def mask_logits(logits, current_cells, masking_data):
    padding_value = config["padding_value"]
    masking_value = config.get("masking_value", float("-inf"))

    allowed = masking_data.eq(1)  # [B, T, 4]
    masked_logits = logits.masked_fill(~allowed, masking_value)

    if current_cells is not None:
        is_pad_cell = current_cells.eq(padding_value)  # [B, T]
        masked_logits = masked_logits.masked_fill(is_pad_cell.unsqueeze(-1), masking_value)

    return masked_logits

def _write_next_token(current_input, b, slot, next_cell, x, y,
                             state_status, goal_cells_xy, context_len):
    
    use_dxdy = bool(config.get("feature_dx_dy",           False))
    use_f6   = bool(config.get("feature_free_neighbours",  False))
    use_f7   = bool(config.get("feature_is_interior",      False))

    fidx_dx        = config.get("fidx_dx")           
    fidx_dy        = config.get("fidx_dy")           
    fidx_free_nbrs = config.get("fidx_free_nbrs")    
    fidx_interior  = config.get("fidx_is_interior")  
    fidx_cell = int(config["fidx_cell_num"])
    fidx_x    = int(config["fidx_x"])
    fidx_y    = int(config["fidx_y"])
    fidx_st   = int(config["fidx_status"])
    current_input[b, slot, fidx_cell] = next_cell
    current_input[b, slot, fidx_x]    = x
    current_input[b, slot, fidx_y]    = y
    current_input[b, slot, fidx_st]   = state_status

    if use_dxdy and fidx_dx is not None and fidx_dy is not None:
        current_input[b, slot, fidx_dx] = goal_cells_xy[b, 0] - x
        current_input[b, slot, fidx_dy] = goal_cells_xy[b, 1] - y

    if use_f6 and fidx_free_nbrs is not None:
        ctx_idx = int(next_cell) - 1
        if 0 <= ctx_idx < context_len:
            current_input[b, slot, fidx_free_nbrs] = current_input[b, ctx_idx, fidx_free_nbrs]

    if use_f7 and fidx_interior is not None:
        ctx_idx = int(next_cell) - 1
        if 0 <= ctx_idx < context_len:
            current_input[b, slot, fidx_interior] = current_input[b, ctx_idx, fidx_interior]

def decode_routes_with_masks(
    *,
    model,
    input_tensor,
    masking_tensor,
    contexts,
    grid_size,
    max_steps=None,
    blank_latent_tail=True,
):
    padding_value  = config["padding_value"]
    context_len    = int(config["context_matrix_length"])
    foundroute_len = int(config["foundroute_matrix_length"])

    if max_steps is None:
        max_steps = foundroute_len

    batch_size = input_tensor.shape[0]
    device     = input_tensor.device

    state_status_maps = [
        {cell.get("Cell_Number", 0): cell.get("State_Status", 0) for cell in contexts[b]}
        for b in range(batch_size)
    ]

    current_input = input_tensor.clone()

    if blank_latent_tail:
        latent_len = current_input.shape[1] - context_len
        if latent_len > 1:
            current_input[:, context_len + 1:, :] = float(padding_value)

    # Feature index constants — no hardcoding
    _fidx_cell = int(config["fidx_cell_num"])
    _fidx_x    = int(config["fidx_x"])
    _fidx_y    = int(config["fidx_y"])
    _fidx_st   = int(config["fidx_status"])
    start_cells = [int(current_input[b, context_len, _fidx_cell].item()) for b in range(batch_size)]

    goal_cells = []
    for b in range(batch_size):
        goal_cell = None
        for i in range(context_len):
            if int(current_input[b, i, _fidx_st].item()) == int(config["status_end"]):
                goal_cell = int(current_input[b, i, _fidx_cell].item())
                break
        goal_cells.append(goal_cell if goal_cell is not None else start_cells[b])

    goal_cells_xy = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
    for b in range(batch_size):
        found = False
        for i in range(context_len):
            if int(current_input[b, i, _fidx_st].item()) == int(config["status_end"]):
                goal_cells_xy[b, 0] = float(current_input[b, i, _fidx_x].item())
                goal_cells_xy[b, 1] = float(current_input[b, i, _fidx_y].item())
                found = True
                break
        if not found:
            goal_cells_xy[b, 0] = float(current_input[b, context_len, _fidx_x].item())
            goal_cells_xy[b, 1] = float(current_input[b, context_len, _fidx_y].item())

    visited_cells      = [set([start_cells[b]]) for b in range(batch_size)]
    done               = [False] * batch_size
    termination_reason = [None]  * batch_size
    hit_horizon        = [True]  * batch_size
    predictions        = []

    for step in range(int(max_steps)):
        dynamic_masks = []
        current_cells = []
        any_active    = False

        for b in range(batch_size):
            current_cell = int(current_input[b, context_len + step, _fidx_cell].item())
            current_cells.append(current_cell)

            if done[b]:
                dynamic_masks.append([0] * config["num_unique_tokens"])
                continue

            any_active = True
            prev_cell  = (int(current_input[b, context_len + step - 1, _fidx_cell].item())
                          if step > 0 else None)

            dyn = build_dynamic_mask(
                current_cell, grid_size, state_status_maps[b], visited_cells[b]
            )

            if bool(config.get("rollout_use_dataset_mask", False)):
                dataset_mask_vec = masking_tensor[b, step].tolist()
                combined = combine_action_masks(
                    dynamic_mask_vec=dyn,
                    dataset_mask_vec=dataset_mask_vec,
                    padding_value=padding_value,
                    num_actions=config["num_unique_tokens"]
                )
            else:
                combined = dyn

            if combined == [0] * config["num_unique_tokens"]:
                done[b]               = True
                termination_reason[b] = "stuck_masked"
                hit_horizon[b]        = False

            dynamic_masks.append(combined)

        mask_step            = torch.tensor(dynamic_masks, dtype=torch.float32,
                                            device=device).unsqueeze(1)
        logits               = model.base_model(current_input)
        logits_step          = logits[:, step:step + 1, :]
        current_cells_tensor = current_input[:, context_len + step, 0].unsqueeze(1)
        masked_logits        = mask_logits(logits_step, current_cells_tensor, mask_step)

        pred = masked_logits.argmax(dim=-1)  # [B, 1]
        for b in range(batch_size):
            if done[b]:
                pred[b, 0] = padding_value

        predictions.append(pred)

        for b in range(batch_size):
            if done[b]:
                continue

            direction = int(pred[b, 0].item())
            if direction == padding_value:
                done[b]               = True
                termination_reason[b] = termination_reason[b] or "unknown_stuck"
                hit_horizon[b]        = False
                continue

            cur_cell  = int(current_cells_tensor[b, 0].item())
            next_info = get_next_cell_info(cur_cell, direction, grid_size, state_status_maps[b])
            if next_info is None:
                done[b]               = True
                termination_reason[b] = "stuck_masked"
                hit_horizon[b]        = False
                continue

            next_cell, x, y, state_status = next_info
            visited_cells[b].add(int(next_cell))

            if int(next_cell) == int(goal_cells[b]):
                done[b]               = True
                termination_reason[b] = "success"
                hit_horizon[b]        = False

            if step < int(max_steps) - 1:
                _write_next_token(
                    current_input, b,
                    slot=context_len + step + 1,
                    next_cell=next_cell, x=x, y=y,
                    state_status=state_status,
                    goal_cells_xy=goal_cells_xy,
                    context_len=context_len,
                )

        if any_active and all(done):
            break

    for b in range(batch_size):
        if termination_reason[b] is None:
            termination_reason[b] = "max_length" if hit_horizon[b] else "unknown_stuck"

    pred_seq = torch.cat(predictions, dim=1).detach().cpu().numpy().tolist()

    pred_cell_paths = []
    for b in range(batch_size):
        dirs      = [d for d in pred_seq[b] if d != padding_value]
        cell_path = directions_to_cell_path(start_cells[b], dirs, grid_size, state_status_maps[b])
        cell_path = _truncate_path_at_goal(cell_path, goal_cells[b])
        pred_cell_paths.append(cell_path)

    return pred_cell_paths, termination_reason

def _truncate_path_at_goal(cell_path, goal_cell):
    if not cell_path or goal_cell is None:
        return cell_path
    for i, c in enumerate(cell_path):
        if int(c) == int(goal_cell):
            return cell_path[:i + 1]
    return cell_path

def build_dynamic_mask(current_cell_num, grid_size, state_status_map, visited_cells):
    n_rows, n_cols = grid_size
    mask = [1] * config["num_unique_tokens"]
    row  = (current_cell_num - 1) // n_cols
    col  = (current_cell_num - 1) % n_cols
    directions = [(row - 1, col), (row + 1, col), (row, col + 1), (row, col - 1)]

    for i, (r, c) in enumerate(directions):
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            mask[i] = 0
            continue
        cell_num = r * n_cols + c + 1
        if state_status_map.get(cell_num, 0) == config["status_obstacle"]:
            mask[i] = 0
            continue
        if cell_num in visited_cells:
            mask[i] = 0
    return mask

def combine_action_masks(dynamic_mask_vec, dataset_mask_vec, padding_value, num_actions):
    if dataset_mask_vec == [padding_value] * num_actions:
        return dynamic_mask_vec
    return [int(m and (d == 1)) for m, d in zip(dynamic_mask_vec, dataset_mask_vec)]

def directions_to_cell_path(start_cell, directions, grid_size, state_status_map):
    path         = [start_cell]
    current_cell = start_cell
    for d in directions:
        next_info = get_next_cell_info(current_cell, d, grid_size, state_status_map)
        if next_info is None:
            break
        next_cell, _, _, _ = next_info
        path.append(next_cell)
        current_cell = next_cell
    return path

def get_next_cell_info(current_cell_num, direction, grid_size, state_status_map):
    n_rows, n_cols = grid_size
    row = (current_cell_num - 1) // n_cols
    col = (current_cell_num - 1) % n_cols
    if direction == 0:
        row -= 1
    elif direction == 1:
        row += 1
    elif direction == 2:
        col += 1
    elif direction == 3:
        col -= 1
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        return None
    next_cell_num = row * n_cols + col + 1
    x             = col + 1
    y             = row + 1
    state_status  = state_status_map.get(next_cell_num, 0)
    return (next_cell_num, x, y, state_status)
