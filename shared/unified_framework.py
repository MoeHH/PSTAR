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
from inference import load_model_from_checkpoint
from visualization import plot_routes

def _visualize_unified_samples(vis_blocked, vis_proceed, classifier_type):
    save_vis    = bool(config.get("save_visualizations", True))
    show_vis    = bool(config.get("show_visualizations", False))
    grid_size   = config["grid_size"]
    results_dir = config.get("results_dir", "results/")
    context_len = int(config["context_matrix_length"])
    padding_val = int(config["padding_value"])
    status_obs  = int(config["status_obstacle"])
    status_end  = int(config["status_end"])

    _ragate_vis = config.get("ragate_visualization_dir", "Results/RAGate/visualization/")
    blocked_dir = os.path.join(_ragate_vis, "blocked_grids")
    proceed_dir = os.path.join(_ragate_vis, "processed_grids")
    if save_vis:
        os.makedirs(blocked_dir, exist_ok=True)
        os.makedirs(proceed_dir, exist_ok=True)

    def _reconstruct_from_tensor(inp):
        fidx_cell_l = int(config["fidx_cell_num"])
        fidx_x_l    = int(config["fidx_x"])
        fidx_y_l    = int(config["fidx_y"])
        fidx_st_l   = int(config["fidx_status"])

        context    = []
        obstacles  = set()
        goal_cell  = None
        start_cell = None
        status_start = int(config["status_start"])

        for ci in range(context_len):
            cn = int(inp[ci, fidx_cell_l].item())
            if cn == padding_val or cn <= 0:
                continue
            x  = int(inp[ci, fidx_x_l].item())
            y  = int(inp[ci, fidx_y_l].item())
            st = int(inp[ci, fidx_st_l].item())
            context.append({
                "Cell_Number":  cn,
                "x":            x,
                "y":            y,
                "State_Status": st,
            })
            if st == status_obs:
                obstacles.add((x, y))
            if st == status_end and goal_cell is None:
                goal_cell = cn
            if st == status_start and start_cell is None:
                start_cell = cn

        # Fallback: latent slot 0 if populated (sol samples)
        if start_cell is None:
            latent_cn = int(inp[context_len, fidx_cell].item())
            if latent_cn != padding_val:
                start_cell = latent_cn

        # Second fallback: first context cell
        if start_cell is None and context:
            start_cell = context[0]["Cell_Number"]

        if start_cell is None:
            start_cell = 1   # hard fallback

        return context, obstacles, start_cell, goal_cell

    def _reconstruct_true_route(found_routes, inp):
        if found_routes:
            cells = []
            for cell in found_routes:
                if isinstance(cell, dict):
                    cn = cell.get("Cell_Number", padding_val)
                else:
                    cn = int(cell)
                if cn != padding_val and cn > 0:
                    cells.append(cn)
            if cells:
                return cells

        _fidx_cell_r = int(config["fidx_cell_num"])
        cells = []
        for li in range(context_len, inp.shape[0]):
            cn = int(inp[li, _fidx_cell_r].item())
            if cn == padding_val:
                break
            cells.append(cn)
        return cells

    # ── Blocked samples ───────────────────────────────────────────────────────
    for entry in vis_blocked:
        inp        = entry["input_tensor"]
        true_route = entry.get("true_route", [])
        outcome    = entry["outcome"]
        prob       = entry["prob"]
        s_idx      = entry["sample_idx"]

        context, obstacles, start_cell, _ = _reconstruct_from_tensor(inp)
        goal_cell = entry.get("goal_cell")
        if goal_cell is None:
            _, _, _, goal_cell = _reconstruct_from_tensor(inp)

        true_cells = _reconstruct_true_route(true_route, inp)
        title = (
            f"UNIFIED Blocked | Sample {s_idx} | "
            f"Classifier: {classifier_type.upper()} | "
            f"Outcome: {outcome} | P(sol)={prob:.4f}"
        )
        save_path = None
        if save_vis:
            save_path = os.path.join(
                blocked_dir, f"sample_{s_idx}_no_route_available.png"
            )

        try:
            plot_routes(
                true_route=true_cells,
                pred_route=[start_cell],  
                grid_size=grid_size,
                start_cell_num=start_cell,
                obstacles=obstacles,
                save_path=save_path,
                show=show_vis,
                title=title,
                context=context,
                intended_goal_cells=goal_cell,
                pred_title=f"{outcome} | P(sol)={prob:.4f}",
                true_route_kind="cells",
                pred_route_kind="cells",
                pred_failure_reason=None,
            )
        except Exception as e:
            print(f"UNIFIED VIS --> Blocked sample {s_idx} failed: {e}")

    # ── Proceed samples ───────────────────────────────────────────────────────
    for entry in vis_proceed:
        inp        = entry["input_tensor"]
        true_route = entry.get("true_route", [])
        pred_route = entry.get("pred_route", [])
        outcome    = entry["outcome"]
        prob       = entry["prob"]
        s_idx      = entry["sample_idx"]
        reached    = entry["goal_reached"]

        context, obstacles, start_cell, _ = _reconstruct_from_tensor(inp)
        goal_cell = entry.get("goal_cell")
        if goal_cell is None:
            _, _, _, goal_cell = _reconstruct_from_tensor(inp)

        true_cells = _reconstruct_true_route(true_route, inp)
        pred_cells = [c for c in pred_route if c != padding_val]
        result_str = "success" if reached else "fail"
        title = (
            f"UNIFIED Proceed | Sample {s_idx} | "
            f"Classifier: {classifier_type.upper()} | "
            f"Outcome: {outcome} | P(sol)={prob:.4f} | Rollout: {result_str}"
        )
        save_path = None
        if save_vis:
            save_path = os.path.join(
                proceed_dir, f"sample_{s_idx}_route_available.png"
            )

        if reached:
            fail_reason = None
        else:
            from route_evaluation import classify_failure_reason
            fail_reason = classify_failure_reason(
                pred_route=pred_cells,
                true_route=true_cells,
                goal_cell=goal_cell,
                context=context,
                grid_size=config["grid_size"],
                max_path_len=int(config["foundroute_matrix_length"]) + 1,
            )
            if fail_reason == "success":
                fail_reason = None

        try:
            plot_routes(
                true_route=true_cells,
                pred_route=pred_cells,
                grid_size=grid_size,
                start_cell_num=start_cell,
                obstacles=obstacles,
                save_path=save_path,
                show=show_vis,
                title=title,
                context=context,
                intended_goal_cells=goal_cell,
                pred_title=f"{outcome} | P(sol)={prob:.4f}",
                true_route_kind="cells",
                pred_route_kind="cells",
                pred_failure_reason=fail_reason,
            )
        except Exception as e:
            print(f"UNIFIED VIS --> Proceed sample {s_idx} failed: {e}")

    n_b = len(vis_blocked)
    n_p = len(vis_proceed)
    if save_vis and (n_b + n_p) > 0:
        print(f"UNIFIED VIS --> Saved {n_b} blocked + {n_p} proceed visualizations "
              f"-> {config.get('ragate_visualization_dir', 'Results/RAGate/visualization/')}")


def _run_rollout_loop(
    model, current_input, state_status_maps_p,
    goal_cells_p, goal_cells_xy_p,
    visited_cells, done_p, goal_reached_p, hit_horizon_p,
    foundroute_len, context_len, padding_value, device,
    rollout_total, rollout_success,
):
    fidx_cell = int(config["fidx_cell_num"])
    fidx_x    = int(config["fidx_x"])
    fidx_y    = int(config["fidx_y"])
    fidx_st   = int(config["fidx_status"])

    P = current_input.shape[0]

    for step in range(foundroute_len):
        dynamic_masks = []
        any_active    = False

        for pi in range(P):
            current_cell = int(current_input[pi, context_len + step, fidx_cell].item())
            if done_p[pi]:
                dynamic_masks.append([0] * config["num_unique_tokens"])
                continue
            any_active = True
            prev_cell  = (int(current_input[pi, context_len + step - 1, fidx_cell].item())
                          if step > 0 else None)

            mask_vec = build_dynamic_mask(
                current_cell, config["grid_size"],
                state_status_maps_p[pi], visited_cells[pi]
            )

            if mask_vec == [0] * config["num_unique_tokens"]:
                done_p[pi]        = True
                hit_horizon_p[pi] = False

            dynamic_masks.append(mask_vec)

        mask_step   = torch.tensor(
            dynamic_masks, dtype=torch.float32, device=device
        ).unsqueeze(1)
        logits      = model.base_model(current_input)
        logits_step = logits[:, step:step + 1, :]
        cur_cells_t = current_input[:, context_len + step, 0].unsqueeze(1)
        ml          = mask_logits(logits_step, cur_cells_t, mask_step)

        pred = ml.argmax(dim=-1)
        for pi in range(P):
            if done_p[pi]:
                pred[pi, 0] = padding_value

        for pi in range(P):
            if done_p[pi]:
                continue
            direction    = int(pred[pi, 0].item())
            current_cell = int(cur_cells_t[pi, 0].item())
            next_info    = get_next_cell_info(
                current_cell, direction,
                config["grid_size"], state_status_maps_p[pi]
            )
            if next_info is None:
                done_p[pi]        = True
                hit_horizon_p[pi] = False
                continue

            next_cell, x, y, state_status = next_info
            visited_cells[pi].add(int(next_cell))

            if int(next_cell) == int(goal_cells_p[pi]):
                done_p[pi]         = True
                goal_reached_p[pi] = True
                hit_horizon_p[pi]  = False



            if step < foundroute_len - 1:
                _write_next_token(
                    current_input, pi,
                    slot=context_len + step + 1,
                    next_cell=next_cell, x=x, y=y,
                    state_status=state_status,
                    goal_cells_xy=goal_cells_xy_p,
                    context_len=context_len,
                )

        if any_active and all(done_p):
            break

    rollout_total   += P
    rollout_success += sum(goal_reached_p)
    return rollout_total, rollout_success, done_p, goal_reached_p


def unified_framework():
    import pickle
    import json
    from torch.utils.data import DataLoader

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"\nUNIFIED --> Device: {device}")

    fidx_cell = int(config["fidx_cell_num"])   # column index for cell number
    fidx_x    = int(config["fidx_x"])          # column index for x coordinate
    fidx_y    = int(config["fidx_y"])          # column index for y coordinate
    fidx_st   = int(config["fidx_status"])     # column index for state_status

    # ── Load transformer (always needed for rollout) ──────────────────────────
    checkpoint_path = os.path.join(
        config["checkpoint_dir"],
        config.get("inference_model_name", config["ce_best_model_name"])
    )
    model = load_model_from_checkpoint(checkpoint_path, device)
    if config.get("ragate_pstar_frozen", True):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("UNIFIED --> PSTAR: FROZEN")
    else:
        model.train()
        for p in model.parameters():
            p.requires_grad = True
        print("UNIFIED --> PSTAR: UNFROZEN")

    # ── Load classifier ───────────────────────────────────────────────────────
    classifier_type = str(config.get("ragate_model", "lda")).strip().lower()
    tau        = float(config["ragate_model_tau1"])
    tau1_active = bool(config.get("ragate_model_tau1_active", True))

    if classifier_type == "lda":
        lda_path = config["ragate_lda_model_path"]
        if not os.path.exists(lda_path):
            raise FileNotFoundError(
                f"UNIFIED --> LDA model not found: {lda_path}\n"
                f"      Run ragate_phase first."
            )
        with open(lda_path, "rb") as f:
            clf_model = pickle.load(f)
        clf_size_kb = os.path.getsize(lda_path) / 1024.0
        print(f"UNIFIED --> LDA classifier loaded: {lda_path}  ({clf_size_kb:.1f} KB)")

    elif classifier_type == "xgboost":
        xgb_path = config["ragate_xgboost_model_path"]
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(
                f"UNIFIED --> XGBoost model not found: {xgb_path}\n"
                f"      Run ragate_phase first."
            )
        with open(xgb_path, "rb") as f:
            clf_model = pickle.load(f)
        try:
            clf_model.get_booster().set_param("device", "cpu")
        except Exception:
            pass
        clf_size_kb = os.path.getsize(xgb_path) / 1024.0
        print(f"UNIFIED --> XGBoost classifier loaded: {xgb_path}  ({clf_size_kb:.1f} KB)")

    elif classifier_type == "mlp":
        from mlp_classifier import RAGateMLPNet
        bin_path = config["ragate_mlp_model_path"]
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"UNIFIED --> Binary model not found: {bin_path}\n"
                f"      Run ragate_phase first."
            )
        ckpt      = torch.load(bin_path, weights_only=False, map_location=device)
        dropout   = float(config.get("ragate_mlp_dropout", 0.1))
        input_dim = int(ckpt.get("input_dim", config["dim"]))
        if "hidden_layers" in ckpt:
            hidden_layers = list(ckpt["hidden_layers"])
        elif "ragate_mlp_hidden_layers" in config:
            hidden_layers = list(config["ragate_mlp_hidden_layers"])
        else:
            raise KeyError(
                "UNIFIED --> Cannot determine MLP hidden_layers — "
                "not found in checkpoint and 'ragate_mlp_hidden_layers' "
                "is missing from config.py."
            )
        clf_net   = RAGateMLPNet(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout).to(device)
        clf_net.load_state_dict(ckpt["model_state_dict"])
        clf_net.eval()
        clf_size_kb = os.path.getsize(bin_path) / 1024.0
        print(f"UNIFIED --> MLP classifier loaded: {bin_path}  ({clf_size_kb:.1f} KB)")
        print(f"UNIFIED --> MLP Architecture: {ckpt.get('arch_str','?')}  input_dim={input_dim}")
    else:
        raise ValueError(
            f"UNIFIED --> Unknown ragate_model: '{classifier_type}'. "
            f"Expected 'lda', 'xgboost', or 'mlp'."
        )

    print(f"UNIFIED --> Threshold: tau={tau} (active={tau1_active})")

    # ── Load RAGate test dataset via shared dataloader ────────────────────
    conf_test_path = config["ragate_test_pt"]
    if not os.path.exists(conf_test_path):
        raise FileNotFoundError(
            f"UNIFIED --> RAGate test dataset not found: {conf_test_path}\n"
            f"      Generate it with the A* dataset-generation repo and copy Datasets/RAGate/ here."
        )
    ragate_loader, ragate_dataset = get_ragate_dataloader(
        conf_test_path, config
    )
    n_total     = len(ragate_dataset)
    print(f"UNIFIED --> RAGate test dataset loaded: {n_total:,} samples")

    emb_test_path    = config.get("ragate_embeddings_test", "")
    embeddings_ready = os.path.exists(emb_test_path)

    if embeddings_ready:
        emb_data = torch.load(emb_test_path, weights_only=False)

        if "X_combined" in emb_data:
            X_all = emb_data["X_combined"]
        elif "X" in emb_data:
            X_all = emb_data["X"]
        else:
            raise KeyError(
                f"UNIFIED --> Embedding file {emb_test_path} has neither 'X_combined' "
                f"nor 'X' key. Re-extract embeddings with extract_features.py."
            )

        y_all = emb_data["y"].tolist()

        expected_dim = None
        if classifier_type == "xgboost":
            try:
                expected_dim = clf_model.get_booster().num_features()
            except Exception:
                pass
        elif classifier_type == "lda":
            try:
                expected_dim = clf_model.coef_.shape[1]
            except Exception:
                pass
        elif classifier_type == "mlp":
            expected_dim = input_dim  # set during binary loading above

        actual_dim = X_all.shape[1]
        if expected_dim is not None and actual_dim != expected_dim:
            raise ValueError(
                f"UNIFIED --> Embedding shape mismatch: disk embeddings have dim={actual_dim} "
                f"but {classifier_type} classifier expects dim={expected_dim}.\n"
                f"      The classifier was trained on different embeddings than those on disk.\n"
                f"      Fix: set ragate_embeddings_force_reextract=True in config.py "
                f"and ensure ragate_use_start_emb, ragate_use_goal_emb, and "
                f"ragate_embedding_all_layers match the settings used to train the classifier."
            )

        print(f"UNIFIED --> Embeddings loaded from disk: {emb_test_path}  "
              f"shape={tuple(X_all.shape)}")
        print(f"UNIFIED --> Transformer forward pass SKIPPED for classification "
              f"(embeddings already extracted)")
    else:
        print(f"UNIFIED --> Embedding file not found: {emb_test_path}")
        print(f"UNIFIED --> Embeddings will be extracted on the fly per batch")
        X_all = None
        y_all = None

    # ── Accumulators ──────────────────────────────────────────────────────────
    context_len    = int(config["context_matrix_length"])
    foundroute_len = int(config["foundroute_matrix_length"])
    padding_value  = int(config["padding_value"])

    CBUG = 0   # Correctly Blocked Unsolvable Grid
    IPUG = 0   # Incorrectly Processed Unsolvable Grid
    WBSG = 0   # Wrongly Blocked Solvable Grid
    CPSG = 0   # Correctly Processed Solvable Grid (pre-rollout)

    n_no_route          = 0
    n_route_available   = 0

    rollout_success = 0
    rollout_total   = 0
    total_solvable   = 0
    total_unsolvable = 0

    emb_batch_size = int(config.get("ragate_embeddings_batch_size", 128))

    # ── Visualization accumulators ────────────────────────────────────────────
    do_vis         = bool(config.get("enable_unified_visualization", False))
    max_vis_blocked = int(config.get("num_unified_visualizations_blocked", 10))
    max_vis_proceed = int(config.get("num_unified_visualizations_proceed", 10))
    vis_blocked    = []   # list of dicts for blocked samples
    vis_proceed    = []   # list of dicts for proceed samples

    # ── Per-sample routing — two paths depending on embedding availability ─────
    if embeddings_ready:
        # PATH A: embeddings from disk — classify all samples first, then rollout
        # Classify in batches using pre-extracted embeddings
        all_probs  = []
        all_labels = y_all

        with torch.no_grad():
            for i in range(0, n_total, emb_batch_size):
                xb    = X_all[i:i + emb_batch_size].to(device)
                if classifier_type == "mlp":
                    p = clf_net(xb).squeeze(1).cpu().numpy()
                else:
                    # lda and xgboost both expose predict_proba
                    p = clf_model.predict_proba(xb.cpu().numpy())[:, 1]
                all_probs.extend(p.tolist())

        # Determine routing for every sample
        proceed_indices_global = []   # indices into ragate_dataset for rollout
        for idx, (prob, label) in enumerate(zip(all_probs, all_labels)):
            total_solvable   += (1 if label == 1 else 0)
            total_unsolvable += (1 if label == 0 else 0)

            if tau1_active and prob < tau:
                n_no_route += 1
                if label == 0: CBUG += 1
                else:          WBSG += 1
                # ── Collect blocked sample for visualization ──────────────
                if do_vis and len(vis_blocked) < max_vis_blocked:
                    s = ragate_dataset.get_raw(idx)
                    ctx = s.get("Context", s.get("context", []))
                    goal_cell = None
                    for cell in ctx:
                        if cell.get("State_Status") == config["status_end"]:
                            goal_cell = int(cell.get("Cell_Number", 0))
                            break
                    true_route = [
                        cell.get("Cell_Number") for cell in s.get("Found_Routes", s.get("found_routes", []))
                        if cell.get("Cell_Number", config["padding_value"])
                           != config["padding_value"]
                    ]
                    vis_blocked.append({
                        "input_tensor": s["input_tensor"],
                        "context":      ctx,
                        "true_route":   true_route,
                        "goal_cell":    goal_cell,
                        "outcome":      "No Route Available",
                        "prob":         float(prob),
                        "sample_idx":   idx,
                    })
            else:
                n_route_available += 1
                proceed_indices_global.append(idx)
                if label == 0: IPUG += 1
                else:          CPSG += 1

        # Rollout only on PROCEED samples
        from torch.utils.data import Dataset as _Dataset

        class _ProceedDataset(_Dataset):
            def __init__(self, samples, indices):
                self._s = samples
                self._i = indices
            def __len__(self):
                return len(self._i)
            def __getitem__(self, pos):
                s = ragate_dataset.get_raw(self._i[pos])
                return s["input_tensor"], int(s["label"])

        proceed_loader = DataLoader(
            _ProceedDataset(ragate_dataset, proceed_indices_global),
            batch_size=emb_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        with torch.no_grad():
            for inp_batch, _ in proceed_loader:
                inp_batch = inp_batch.to(device)
                P         = inp_batch.shape[0]

                state_status_maps_p = []
                for b in range(P):
                    ssm = {}
                    for ci in range(context_len):
                        cn = int(inp_batch[b, ci, fidx_cell].item())
                        st = int(inp_batch[b, ci, fidx_st].item())
                        ssm[cn] = st
                    state_status_maps_p.append(ssm)

                goal_cells_p    = []
                goal_cells_xy_p = torch.zeros(P, 2, device=device, dtype=torch.float32)
                for b in range(P):
                    goal_cell = goal_x = goal_y = None
                    for ci in range(context_len):
                        if int(inp_batch[b, ci, fidx_st].item()) == int(config["status_end"]):
                            goal_cell = int(inp_batch[b, ci, fidx_cell].item())
                            goal_x    = float(inp_batch[b, ci, fidx_x].item())
                            goal_y    = float(inp_batch[b, ci, fidx_y].item())
                            break
                    if goal_cell is None:
                        goal_cell = int(inp_batch[b, context_len, fidx_cell].item())
                        goal_x    = float(inp_batch[b, context_len, fidx_x].item())
                        goal_y    = float(inp_batch[b, context_len, fidx_y].item())
                    goal_cells_p.append(goal_cell)
                    goal_cells_xy_p[b, 0] = goal_x
                    goal_cells_xy_p[b, 1] = goal_y



                current_input = inp_batch.clone()
                visited_cells = [set() for _ in range(P)]
                for b in range(P):
                    visited_cells[b].add(int(current_input[b, context_len, fidx_cell].item()))

                done_p         = [False] * P
                goal_reached_p = [False] * P
                hit_horizon_p  = [True]  * P

                latent_len = current_input.shape[1] - context_len
                if latent_len > 1:
                    current_input[:, context_len + 1:, :] = float(config["padding_value"])

                rollout_total, rollout_success, done_p, goal_reached_p = \
                    _run_rollout_loop(
                        model, current_input, state_status_maps_p,
                        goal_cells_p, goal_cells_xy_p,
                        visited_cells, done_p, goal_reached_p, hit_horizon_p,
                        foundroute_len, context_len, padding_value, device,
                        rollout_total, rollout_success,
                    )

                # ── Collect proceed samples for visualization (PATH A) ─────
                if do_vis and len(vis_proceed) < max_vis_proceed:
                    batch_offset = rollout_total - P

                    for pi in range(P):
                        if len(vis_proceed) >= max_vis_proceed:
                            break
                        raw_idx = proceed_indices_global[batch_offset + pi]
                        s       = ragate_dataset.get_raw(raw_idx)
                        ctx     = s.get("Context", s.get("context", []))
                        outcome = "Route Available"
                        true_route = [
                            cell.get("Cell_Number")
                            for cell in s.get("Found_Routes", s.get("found_routes", []))
                            if cell.get("Cell_Number", config["padding_value"])
                               != config["padding_value"]
                        ]
                        pred_cells = [int(inp_batch[pi, context_len, fidx_cell].item())]
                        for step in range(foundroute_len - 1):
                            nc = int(current_input[pi, context_len + step + 1, fidx_cell].item())
                            if nc == int(config["padding_value"]):
                                break
                            pred_cells.append(nc)
                            if nc == goal_cells_p[pi]:
                                break
                        vis_proceed.append({
                            "input_tensor": inp_batch[pi].cpu(),
                            "context":      ctx,
                            "true_route":   true_route,
                            "pred_route":   pred_cells,
                            "goal_cell":    goal_cells_p[pi],
                            "outcome":      outcome,
                            "prob":         float(all_probs[raw_idx]),
                            "sample_idx":   raw_idx,
                            "goal_reached": goal_reached_p[pi],
                        })

    else:
        from torch.utils.data import Dataset as _Dataset

        class _ConfDataset(_Dataset):
            def __init__(self, samples):
                self._s = samples
            def __len__(self):
                return len(self._s)
            def __getitem__(self, idx):
                s = ragate_dataset.get_raw(idx)
                return s["input_tensor"], int(s["label"])

        loader = DataLoader(
            _ConfDataset(ragate_dataset),
            batch_size=emb_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        with torch.no_grad():
            for inp_batch, label_batch in loader:
                inp_batch = inp_batch.to(device)
                labels    = label_batch.tolist()
                B         = inp_batch.shape[0]

                emb_dict = model.get_context_embeddings(inp_batch)

                parts = []
                if config.get("ragate_use_mean_emb", True) and "context_mean" in emb_dict:
                    parts.append(emb_dict["context_mean"])
                if config.get("ragate_use_start_emb", False) and "start_emb" in emb_dict:
                    parts.append(emb_dict["start_emb"])
                if config.get("ragate_use_goal_emb", False) and "goal_emb" in emb_dict:
                    parts.append(emb_dict["goal_emb"])

                if not parts:
                    raise ValueError(
                        "UNIFIED --> No embedding parts available. "
                        "Ensure ragate_use_mean_emb=True in config."
                    )
                X_combined = torch.cat(parts, dim=1)  # [B, combined_dim]

                if classifier_type == "mlp":
                    probs = clf_net(X_combined).squeeze(1).cpu().numpy()
                else:
                    # lda and xgboost both expose predict_proba
                    probs = clf_model.predict_proba(X_combined.cpu().numpy())[:, 1]

                proceed_mask = []
                batch_probs  = probs.tolist()   # save for vis lookup

                for i, prob in enumerate(probs):
                    label = labels[i]
                    total_solvable   += (1 if label == 1 else 0)
                    total_unsolvable += (1 if label == 0 else 0)

                    if tau1_active and prob < tau:
                        n_no_route += 1
                        proceed_mask.append(False)
                        if label == 0: CBUG += 1
                        else:          WBSG += 1
                        # ── Collect blocked sample for visualization ──────
                        if do_vis and len(vis_blocked) < max_vis_blocked:
                            # global sample index = batch_idx * batch + i
                            g_idx = batch_idx * emb_batch_size + i
                            s     = ragate_dataset.get_raw(g_idx) \
                                    if g_idx < n_total else None
                            if s is not None:
                                ctx = s.get("Context", s.get("context", []))
                                goal_cell = None
                                for cell in ctx:
                                    if cell.get("State_Status") == \
                                       config["status_end"]:
                                        goal_cell = int(
                                            cell.get("Cell_Number", 0))
                                        break
                                true_route = [
                                    cell.get("Cell_Number")
                                    for cell in s.get("Found_Routes", s.get("found_routes", []))
                                    if cell.get("Cell_Number",
                                               config["padding_value"])
                                       != config["padding_value"]
                                ]
                                vis_blocked.append({
                                    "input_tensor": inp_batch[i].cpu(),
                                    "context":      ctx,
                                    "true_route":   true_route,
                                    "goal_cell":    goal_cell,
                                    "outcome":      "No Route Available",
                                    "prob":         float(prob),
                                    "sample_idx":   g_idx,
                                })
                    else:
                        n_route_available += 1
                        proceed_mask.append(True)
                        if label == 0: IPUG += 1
                        else:          CPSG += 1

                proceed_indices = [i for i, p in enumerate(proceed_mask) if p]
                if not proceed_indices:
                    continue

                inp_proceed = inp_batch[proceed_indices]
                P           = inp_proceed.shape[0]

                state_status_maps_p = []
                for i in proceed_indices:
                    ssm = {}
                    for ci in range(context_len):
                        cn = int(inp_batch[i, ci, fidx_cell].item())
                        st = int(inp_batch[i, ci, fidx_st].item())
                        ssm[cn] = st
                    state_status_maps_p.append(ssm)

                goal_cells_p    = []
                goal_cells_xy_p = torch.zeros(P, 2, device=device, dtype=torch.float32)
                for pi, i in enumerate(proceed_indices):
                    goal_cell = goal_x = goal_y = None
                    for ci in range(context_len):
                        if int(inp_batch[i, ci, fidx_st].item()) == int(config["status_end"]):
                            goal_cell = int(inp_batch[i, ci, fidx_cell].item())
                            goal_x    = float(inp_batch[i, ci, 1].item())
                            goal_y    = float(inp_batch[i, ci, 2].item())
                            break
                    if goal_cell is None:
                        goal_cell = int(inp_batch[i, context_len, 0].item())
                        goal_x    = float(inp_batch[i, context_len, 1].item())
                        goal_y    = float(inp_batch[i, context_len, 2].item())
                    goal_cells_p.append(goal_cell)
                    goal_cells_xy_p[pi, 0] = goal_x
                    goal_cells_xy_p[pi, 1] = goal_y

                current_input = inp_proceed.clone()
                visited_cells = [set() for _ in range(P)]
                for pi in range(P):
                    visited_cells[pi].add(int(current_input[pi, context_len, 0].item()))

                done_p         = [False] * P
                goal_reached_p = [False] * P
                hit_horizon_p  = [True]  * P

                latent_len = current_input.shape[1] - context_len
                if latent_len > 1:
                    current_input[:, context_len + 1:, :] = float(config["padding_value"])

                rollout_total, rollout_success, done_p, goal_reached_p = \
                    _run_rollout_loop(
                        model, current_input, state_status_maps_p,
                        goal_cells_p, goal_cells_xy_p,
                        visited_cells, done_p, goal_reached_p, hit_horizon_p,
                        foundroute_len, context_len, padding_value, device,
                        rollout_total, rollout_success,
                    )

                # ── Collect proceed samples for visualization (PATH B) ─────
                if do_vis and len(vis_proceed) < max_vis_proceed:
                    batch_offset = rollout_total - P
                    for pi, orig_i in enumerate(proceed_indices):
                        if len(vis_proceed) >= max_vis_proceed:
                            break
                        g_idx   = batch_idx * emb_batch_size + orig_i
                        s       = ragate_dataset.get_raw(g_idx) \
                                  if g_idx < n_total else None
                        if s is None:
                            continue
                        ctx     = s.get("Context", s.get("context", []))
                        prob    = batch_probs[orig_i]
                        outcome = "Route Available"
                        true_route = [
                            cell.get("Cell_Number")
                            for cell in s.get("Found_Routes", s.get("found_routes", []))
                            if cell.get("Cell_Number", config["padding_value"])
                               != config["padding_value"]
                        ]
                        pred_cells = [int(current_input[pi, context_len,
                                                        0].item())]
                        for step in range(foundroute_len - 1):
                            nc = int(current_input[pi, context_len + step + 1,
                                                   0].item())
                            if nc == int(config["padding_value"]):
                                break
                            pred_cells.append(nc)
                            if nc == goal_cells_p[pi]:
                                break
                        vis_proceed.append({
                            "input_tensor": inp_batch[orig_i].cpu(),
                            "context":      ctx,
                            "true_route":   true_route,
                            "pred_route":   pred_cells,
                            "goal_cell":    goal_cells_p[pi],
                            "outcome":      outcome,
                            "prob":         float(prob),
                            "sample_idx":   g_idx,
                            "goal_reached": goal_reached_p[pi],
                        })

    # ── Compute final metrics ─────────────────────────────────────────────────
    cbug_rate = CBUG / max(CBUG + IPUG, 1) * 100.0
    block_pur = CBUG / max(CBUG + WBSG, 1) * 100.0
    f1_pct    = (2.0 * (block_pur / 100.0) * (cbug_rate / 100.0)
                 / max((block_pur / 100.0) + (cbug_rate / 100.0), 1e-9)) * 100.0
    cpsg_rate = CPSG / max(total_solvable, 1)  * 100.0
    wbsg_rate = WBSG / max(total_solvable, 1)  * 100.0
    rollout_success_rate = rollout_success / max(rollout_total, 1) * 100.0

    # ── Print report ──────────────────────────────────────────────────────────
    emb_source = "disk" if embeddings_ready else "on-the-fly"
    W = 72
    print()
    print("═" * W)
    print(f"  UNIFIED INFERENCE WITH CLASSIFIER — Results")
    print(f"  Classifier: {classifier_type.upper()}  |  "
          f"tau={tau} (active={tau1_active})")
    print(f"  Embeddings: {emb_source}")
    print(f"  Dataset: {conf_test_path}")
    print(f"  N={n_total:,}  |  Solvable={total_solvable:,}  |  "
          f"Unsolvable={total_unsolvable:,}")
    print("─" * W)
    print(f"  Classifier Outcome Distribution:")
    print(f"    No Route Available (BLOCKED):  {n_no_route:,}")
    print(f"    Route Available    (PROCEED):  {n_route_available:,}")
    print("─" * W)
    print(f"  Gate Outcome Counts:")
    print(f"    Correctly Blocked Unsolvable Grid     (CBUG): {CBUG:,}")
    print(f"    Incorrectly Processed Unsolvable Grid (IPUG): {IPUG:,}")
    print(f"    Wrongly Blocked Solvable Grid         (WBSG): {WBSG:,}")
    print(f"    Correctly Processed Solvable Grid     (CPSG): {CPSG:,}")
    print("─" * W)
    print(f"  Gate Metrics:")
    print(f"    CBUG Rate  (unsolvable correctly blocked):  {cbug_rate:.2f}%")
    print(f"    CPSG Rate  (solvable correctly processed):  {cpsg_rate:.2f}%")
    print(f"    WBSG Rate  (solvable wrongly blocked):      {wbsg_rate:.2f}%"
          f"  ← primary gate")
    print(f"    Classifier Score (F1):                      {f1_pct:.2f}%"
          f"  ← gate quality")
    print("─" * W)
    print(f"  Rollout Results (PROCEED samples only):")
    print(f"    Samples sent to rollout:  {rollout_total:,}")
    print(f"    Rollout successes:        {rollout_success:,}")
    print(f"    Rollout success rate:     {rollout_success_rate:.2f}%")
    print("═" * W)
    print()

    # ── Visualization ─────────────────────────────────────────────────────────
    if do_vis and (vis_blocked or vis_proceed):
        _visualize_unified_samples(vis_blocked, vis_proceed, classifier_type)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "classifier":        classifier_type,
        "tau":              tau,
        "tau1_active":       tau1_active,

        "embeddings_source": emb_source,
        "dataset":           conf_test_path,
        "total_samples":     n_total,
        "total_solvable":    total_solvable,
        "total_unsolvable":  total_unsolvable,
        "outcome_distribution": {
            "No Route Available":          n_no_route,

        },
        "gate_counts": {
            "CBUG": CBUG,
            "IPUG": IPUG,
            "WBSG": WBSG,
            "CPSG": CPSG,
        },
        "gate_metrics": {
            "cbug_rate_pct":  round(cbug_rate, 2),
            "cpsg_rate_pct":  round(cpsg_rate, 2),
            "wbsg_rate_pct":  round(wbsg_rate, 2),
            "f1_pct":         round(f1_pct,    2),
        },
        "rollout": {
            "total_proceed":         rollout_total,
            "rollout_successes":     rollout_success,
            "rollout_success_rate_pct": round(rollout_success_rate, 2),
        },
    }

    out_path = config["inference_classifier_results_output"]
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"--> Results saved to {out_path}")


if __name__ == "__main__":
    inference_main()
