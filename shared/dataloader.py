import os
import mmap
import json
import pickle
import ijson
import torch
from torch.utils.data import Dataset, DataLoader
from config import config

def _build_json_offset_index(file_path):
    import re
    offsets = []
    with open(file_path, "rb") as f:
        raw = f.read()
    depth      = 0
    in_string  = False
    escape     = False
    item_start = None
    i          = 0
    while i < len(raw):
        b = raw[i:i+1]
        if escape:
            escape = False
        elif in_string:
            if b == b'\\':
                escape = True
            elif b == b'"':
                in_string = False
        else:
            if b == b'"':
                in_string = True
            elif b == b'{':
                if depth == 1:
                    item_start = i
                depth += 1
            elif b == b'}':
                depth -= 1
                if depth == 1 and item_start is not None:
                    offsets.append(item_start)
                    item_start = None
            elif b == b'[':
                depth += 1
            elif b == b']':
                depth -= 1
        i += 1
    return offsets

_META_KEYS = ("Template_ID", "Open_Percent", "Pair_ID", "Direction", "Grid_Size", "Split")

def _probe_indices(n):
    return sorted({0, n // 2, n - 1}) if n > 0 else []

def check_sample_against_config(sample, source, cfg=None, check_tensor_width=None):
    cfg = cfg or config
    n_rows, n_cols = (int(v) for v in cfg["grid_size"])
    n_cells  = n_rows * n_cols
    ctx_len  = int(cfg["context_matrix_length"])
    lat_len  = int(cfg["foundroute_matrix_length"])
    tgt_len  = int(cfg["target_matrix_length"])
    seq_len  = int(cfg["sequence_len"])
    pad      = cfg["padding_value"]
    problems = []

    gs = sample.get("Grid_Size")
    if gs is not None and [int(v) for v in gs] != [n_rows, n_cols]:
        problems.append(f"sample Grid_Size={list(gs)} but config grid_size={[n_rows, n_cols]}")

    ctx = sample.get("Context", sample.get("context"))
    if isinstance(ctx, list) and ctx:
        real = [c for c in ctx if isinstance(c, dict) and c.get("Cell_Number", pad) != pad]
        if len(real) != n_cells:
            problems.append(f"{len(real)} real context cells, expected rows*cols={n_cells}")
        if real:
            max_cn = max(int(c["Cell_Number"]) for c in real)
            max_x  = max(int(c.get("x", 0)) for c in real)
            max_y  = max(int(c.get("y", 0)) for c in real)
            if max_cn > n_cells or max_x > n_cols or max_y > n_rows:
                problems.append(f"context cell/coord out of range "
                                f"(max cell={max_cn}, x={max_x}, y={max_y}) for {n_rows}x{n_cols}")

    fr = sample.get("Found_Routes", sample.get("found_routes"))
    if isinstance(fr, list):
        n_route = sum(1 for c in fr if isinstance(c, dict) and c.get("Cell_Number", pad) != pad)
        if n_route > lat_len:
            problems.append(f"route has {n_route} cells > foundroute_matrix_length={lat_len}")

    it = sample.get("input_tensor")
    if it is not None:
        if int(it.shape[0]) != seq_len:
            problems.append(f"input_tensor length {int(it.shape[0])} != sequence_len={seq_len} "
                            f"(context {ctx_len} + latent {lat_len})")
        if check_tensor_width is not None and int(it.shape[-1]) < int(check_tensor_width):
            problems.append(f"input_tensor width {int(it.shape[-1])} < feature_size={check_tensor_width}")
    for key, want in (("masking_tensor", lat_len), ("target_tensor", tgt_len)):
        t = sample.get(key)
        if t is not None and int(t.shape[0]) != want:
            problems.append(f"{key} length {int(t.shape[0])} != {want}")

    if problems:
        raise ValueError(
            f"[DATALOADER] Dataset does not match config ({source}):\n  - "
            + "\n  - ".join(problems)
            + "\n  Fix config grid_size / foundroute_matrix_length, or point the "
              "dataset paths at the matching Datasets/<RxC>/ folder.")

def _cell_cols():
    return [int(config[k]) for k in ("fidx_cell_num", "fidx_x", "fidx_y", "fidx_status")]

def reconstruct_context(input_tensor):
    ctx_len, pad = int(config["context_matrix_length"]), float(config["padding_value"])
    rows = input_tensor[:ctx_len, _cell_cols()].tolist()
    return [{"Cell_Number": int(cn), "x": int(x), "y": int(y), "State_Status": int(st)}
            for cn, x, y, st in rows if cn != pad and cn > 0]

def reconstruct_found_routes(input_tensor):
    """Route cell dicts from the latent slice of input_tensor (stops at padding)."""
    ctx_len, pad = int(config["context_matrix_length"]), float(config["padding_value"])
    out = []
    for cn, x, y, st in input_tensor[ctx_len:, _cell_cols()].tolist():
        if cn == pad or cn <= 0:
            break
        out.append({"Cell_Number": int(cn), "x": int(x), "y": int(y), "State_Status": int(st)})
    return out

def raw_context(sample):
    ctx = sample.get("Context", sample.get("context"))
    return ctx if ctx is not None else reconstruct_context(sample["input_tensor"])

def raw_found_routes(sample):
    fr = sample.get("Found_Routes", sample.get("found_routes"))
    if fr is not None:
        return fr
    if int(sample.get("label", 1)) == 0 or int(sample.get("TourCost", 1)) == 0:
        return []
    return reconstruct_found_routes(sample["input_tensor"])

def get_sample_metadata(dataset, idx):
    ds = getattr(dataset, "dataset", dataset)        
    raw = None
    if hasattr(ds, "get_raw"):
        raw = ds.get_raw(idx)
    elif getattr(ds, "_use_pt", False):
        raw = ds._pt_dataset._samples[idx]
    elif isinstance(getattr(ds, "samples", None), _StreamingPlaceholder):
        raw = ds.samples.read_at(ds._offsets[idx])
    elif hasattr(ds, "samples"):
        raw = ds.samples[idx]
    elif hasattr(ds, "_samples"):
        raw = ds._samples[idx]
    if not isinstance(raw, dict):
        return {}
    return {k: raw[k] for k in _META_KEYS if k in raw}

class _StreamingPlaceholder:
    def __init__(self, count, filepath):
        self._count    = count
        self._filepath = filepath
        self._file     = open(filepath, "rb")
        self._mmap     = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self._count

    def __del__(self):
        try:
            if hasattr(self, "_mmap") and self._mmap:
                self._mmap.close()
            if hasattr(self, "_file") and self._file:
                self._file.close()
        except Exception:
            pass

    def read_at(self, offset):
        mm    = self._mmap
        depth = 0
        in_string = False
        escape    = False
        i         = offset
        n         = len(mm)
        while i < n:
            b = mm[i:i+1]
            if escape:
                escape = False
            elif in_string:
                if b == b"\\":
                    escape = True
                elif b == b'"':
                    in_string = False
            else:
                if b == b'"':
                    in_string = True
                elif b in (b"{", b"["):
                    depth += 1
                elif b in (b"}", b"]"):
                    depth -= 1
                    if depth == 0:
                        return json.loads(mm[offset:i+1])
            i += 1
        raise ValueError(f"Incomplete JSON object at offset {offset}")

class TensorPtDataset(Dataset):
    def __init__(self, pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Pre-processed .pt file not found: {pt_path}\n"
                f"Run: python preprocess_dataset.py"
            )
        print(f"--> Loading tensor dataset: {pt_path}")
        self._samples = torch.load(pt_path, weights_only=False)
        print(f"<-- Loaded {len(self._samples):,} samples from {pt_path}\n")
        for _i in _probe_indices(len(self._samples)):
            check_sample_against_config(self._samples[_i], pt_path)
        use_dxdy = bool(config.get("feature_dx_dy", False))
        use_f6   = bool(config.get("feature_free_neighbours", False))
        use_f7   = bool(config.get("feature_is_interior", False))
        col_idx = [0, 1, 2, 3]
        if use_dxdy:
            col_idx += [4, 5]
        if use_f6:
            col_idx += [6]
        if use_f7:
            col_idx += [7]

        if self._samples and "input_tensor" in self._samples[0]:
            pt_width = int(self._samples[0]["input_tensor"].shape[-1])
            need     = max(col_idx) + 1
            if pt_width < need:
                raise ValueError(
                    f"[DATALOADER] .pt file has feature_size={pt_width}, but the "
                    f"active feature toggles need column {max(col_idx)} "
                    f"(feature_size={len(col_idx)}).\n"
                    f"Regenerate the dataset with all 8 features (A* generator), "
                    f"or turn the extra feature toggles off."
                )
            if pt_width != len(col_idx):
                _cols = torch.as_tensor(col_idx, dtype=torch.long)
                for s in self._samples:
                    it = s.get("input_tensor")
                    if it is not None and it.shape[-1] >= need:
                        s["input_tensor"] = it.index_select(-1, _cols)
                print(f"--> Selected feature columns {col_idx} "
                      f"(feature_size={len(col_idx)}) from {pt_width}-wide tensors\n")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        s            = self._samples[idx]
        self._current = s
        input_tensor = s["input_tensor"]
        context = s.get("Context", s.get("context", None))
        if context is None:
            context = self._reconstruct_context(input_tensor)

        found_routes = s.get("Found_Routes", s.get("found_routes", None))
        if found_routes is None:
            found_routes = self._reconstruct_found_routes(input_tensor)

        return (
            input_tensor,
            s["masking_tensor"],
            s["target_tensor"],
            context,
            found_routes,
        )

    def _reconstruct_context(self, input_tensor):
        return reconstruct_context(input_tensor)

    def _reconstruct_context_legacy(self, input_tensor):
        from config import config as _cfg
        context_len  = int(_cfg["context_matrix_length"])
        padding_val  = float(_cfg["padding_value"])
        fidx_cell    = int(_cfg["fidx_cell_num"])
        fidx_x       = int(_cfg["fidx_x"])
        fidx_y       = int(_cfg["fidx_y"])
        fidx_st      = int(_cfg["fidx_status"])
        context = []
        for i in range(context_len):
            cn = input_tensor[i, fidx_cell].item()
            if cn == padding_val or cn <= 0:
                continue
            context.append({
                "Cell_Number":  int(cn),
                "x":            int(input_tensor[i, fidx_x].item()),
                "y":            int(input_tensor[i, fidx_y].item()),
                "State_Status": int(input_tensor[i, fidx_st].item()),
            })
        return context

    def _reconstruct_found_routes(self, input_tensor):
        if int(self._current.get("TourCost", 1)) == 0:   # nosol: no route
            return []
        return reconstruct_found_routes(input_tensor)

    def _reconstruct_found_routes_legacy(self, input_tensor):
        from config import config as _cfg
        context_len = int(_cfg["context_matrix_length"])
        seq_len     = input_tensor.shape[0]
        padding_val = float(_cfg["padding_value"])
        fidx_cell   = int(_cfg["fidx_cell_num"])
        fidx_x      = int(_cfg["fidx_x"])
        fidx_y      = int(_cfg["fidx_y"])
        fidx_st     = int(_cfg["fidx_status"])
        found_routes = []
        for i in range(context_len, seq_len):
            cn = input_tensor[i, fidx_cell].item()
            if cn == padding_val or cn <= 0:
                break
            found_routes.append({
                "Cell_Number":  int(cn),
                "x":            int(input_tensor[i, fidx_x].item()),
                "y":            int(input_tensor[i, fidx_y].item()),
                "State_Status": int(input_tensor[i, fidx_st].item()),
            })
        return found_routes


class PathfindingDataset(Dataset):
    def __init__(self, file_path, config, pickle_key=None):
        self.config    = config
        self.file_path = file_path
        self.samples   = []
        file_format    = config.get("file_format", "pickle")
        data_streaming = bool(config.get("data_streaming", False))
        pickle_path    = config.get(pickle_key) if pickle_key else file_path.replace(".json", ".pkl")

        pt_path = file_path.replace(".json", ".pt")
        if file_format == "tensor_pt" and os.path.exists(pt_path):
            self._pt_dataset = TensorPtDataset(pt_path)
            self.samples     = self._pt_dataset._samples
            self._use_pt     = True

        elif file_format == "pickle" and os.path.exists(pickle_path):
            print(f"--> Loading pickle file: {pickle_path}")
            with open(pickle_path, "rb") as f:
                self.samples = pickle.load(f)
            print(f"<-- DataLoaded {len(self.samples)} samples from {pickle_path} (pickle) into memory\n")
            self._use_pt = False

        elif data_streaming:
            idx_path = file_path + ".idx"
            if os.path.exists(idx_path):
                print(f"--> Loading offset index: {idx_path}")
                with open(idx_path, "rb") as _f:
                    self._offsets = pickle.load(_f)
                print(f"<-- Offset index loaded: {len(self._offsets)} samples\n")
            else:
                print(f"--> Building offset index for: {file_path}")
                print(f"    (one-time scan — saved to {idx_path} for future runs)")
                self._offsets = _build_json_offset_index(file_path)
                with open(idx_path, "wb") as _f:
                    pickle.dump(self._offsets, _f)
                print(f"<-- Offset index built: {len(self._offsets)} samples\n")
            self.samples = _StreamingPlaceholder(len(self._offsets), file_path)

        else:
            print(f"--> Loading JSON file: {file_path}")
            with open(file_path, "r") as f:
                for sample in ijson.items(f, "item"):
                    self.samples.append(sample)
            print(f"<-- DataLoaded {len(self.samples)} samples from {file_path} (JSON) into memory\n")

        self.length                  = len(self.samples)
        self.grid_size               = config["grid_size"]
        self.feature_size            = config["feature_size"]
        self.context_matrix_length   = config["context_matrix_length"]
        self.foundroute_matrix_length = config["foundroute_matrix_length"]
        self.target_matrix_length    = config["target_matrix_length"]
        self.padding_value           = config["padding_value"]

        if not getattr(self, "_use_pt", False):
            for _i in _probe_indices(self.length):
                _raw = (self.samples.read_at(self._offsets[_i])
                        if isinstance(self.samples, _StreamingPlaceholder)
                        else self.samples[_i])
                check_sample_against_config(_raw, file_path)

        if bool(self.config.get("dataset_integrity_check", False)):
            self._validate_samples()

    def _validate_samples(self):
        sample_size   = int(self.config.get("dataset_integrity_sample_size", 5))
        sample_size   = max(1, min(sample_size, self.length))
        padding_value = self.padding_value

        empty_found_routes = 0
        empty_targets      = 0

        for i in range(sample_size):
            sample       = self.samples[i]
            found_routes = sample.get("Found_Routes", [])
            targets      = sample.get("Targets", [])

            has_route = any(
                cell.get("Cell_Number", padding_value) != padding_value
                for cell in found_routes
                if isinstance(cell, dict)
            )
            if not has_route:
                empty_found_routes += 1

            has_target = any(
                t.get("target_vec") is not None and
                not all(v == padding_value for v in t.get("target_vec", []))
                for t in targets
                if isinstance(t, dict)
            )
            if not has_target:
                empty_targets += 1

        print(
            f"[DATASET CHECK] samples={sample_size} "
            f"empty_found_routes={empty_found_routes} "
            f"empty_targets={empty_targets}"
        )

    def __len__(self):
        return self.length

    def _build_feature_vec(self, cell_num, x, y, status,
                           dx_to_goal, dy_to_goal, f6, f7,
                           pad):
        """
        Return a Python list of floats with length == self.feature_size,
        respecting the active feature toggles from config.
        """
        use_dxdy = bool(self.config.get("feature_dx_dy",           False))
        use_f6   = bool(self.config.get("feature_free_neighbours",  False))
        use_f7   = bool(self.config.get("feature_is_interior",      False))

        vec = [float(cell_num), float(x), float(y), float(status)]  
        if use_dxdy:
            vec.append(float(dx_to_goal))  # f4
            vec.append(float(dy_to_goal))  # f5
        if use_f6:
            vec.append(float(f6))          # f6
        if use_f7:
            vec.append(float(f7))          # f7
        return vec

    def __getitem__(self, idx):
        if getattr(self, "_use_pt", False):
            return self._pt_dataset[idx]
        if isinstance(self.samples, _StreamingPlaceholder):
            offset = self._offsets[idx]
            sample = self.samples.read_at(offset)
        else:
            sample = self.samples[idx]

        context      = sample.get('Context', [])
        found_routes = sample.get('Found_Routes', [])
        target       = sample.get('Targets', [])
        masking_data = sample.get('Masking', [])

        pad           = float(self.padding_value)
        padding_value = self.padding_value
        goal_x, goal_y = None, None
        status_end     = self.config["status_end"]

        for cell in context:
            cell_num = cell.get('Cell_Number', padding_value)
            x        = cell.get('x', padding_value)
            y        = cell.get('y', padding_value)
            if cell_num == padding_value or x == padding_value or y == padding_value:
                continue
            if cell.get('State_Status') == status_end:
                goal_x = x
                goal_y = y
                break

        if goal_x is None or goal_y is None:
            for cell in context:
                cell_num = cell.get('Cell_Number', padding_value)
                x        = cell.get('x', padding_value)
                y        = cell.get('y', padding_value)
                if cell_num == padding_value or x == padding_value or y == padding_value:
                    continue
                goal_x = x
                goal_y = y
                break

        if goal_x is None or goal_y is None:
            goal_x = 1
            goal_y = 1
        use_dxdy = bool(self.config.get("feature_dx_dy",           False))
        use_f6   = bool(self.config.get("feature_free_neighbours",  False))
        use_f7   = bool(self.config.get("feature_is_interior",      False))

        n_rows, n_cols  = self.grid_size
        max_neighbours  = float(self.config.get("max_neighbours", 4))
        status_obstacle = self.config["status_obstacle"]

        obstacle_cells = set()
        if use_f6 or use_f7:
            for cell in context:
                if cell.get("State_Status") == status_obstacle:
                    cn = cell.get("Cell_Number", padding_value)
                    if cn != padding_value:
                        obstacle_cells.add(cn)

        def _compute_f6_f7(cell_num):
            if cell_num == padding_value:
                return 0.0, 0.0
            row = (cell_num - 1) // n_cols
            col = (cell_num - 1) % n_cols
            free_count = 0
            has_oob    = False
            for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                r, c = row + dr, col + dc
                if not (0 <= r < n_rows and 0 <= c < n_cols):
                    has_oob = True
                elif r * n_cols + c + 1 not in obstacle_cells:
                    free_count += 1
            f6 = (free_count / max_neighbours) if use_f6 else 0.0
            f7 = (0.0 if has_oob else 1.0)    if use_f7 else 0.0
            return f6, f7

        cell_features_67 = {}
        if use_f6 or use_f7:
            for cell in context:
                cn = cell.get("Cell_Number", padding_value)
                if cn == padding_value:
                    continue
                if ("free_neighbours" in cell) or ("is_interior" in cell):
                    f6 = float(cell.get("free_neighbours", 0.0)) if use_f6 else 0.0
                    f7 = float(cell.get("is_interior", 0.0))     if use_f7 else 0.0
                    cell_features_67[cn] = (f6, f7)
                else:
                    cell_features_67[cn] = _compute_f6_f7(cn)

        context_matrix = torch.full(
            (self.context_matrix_length, self.feature_size), pad, dtype=torch.float32
        )
        for i, cell in enumerate(context):
            if i >= self.context_matrix_length:
                break
            cell_num = cell.get('Cell_Number', padding_value)
            x        = cell.get('x', padding_value)
            y        = cell.get('y', padding_value)
            status   = cell.get('State_Status', self.config["status_free"])

            is_pad_cell = (cell_num == padding_value or
                           x == padding_value or y == padding_value)
            if not is_pad_cell:
                dx_to_goal = float(cell.get("dx_to_goal", goal_x - x)) if use_dxdy else 0.0
                dy_to_goal = float(cell.get("dy_to_goal", goal_y - y)) if use_dxdy else 0.0
            else:
                dx_to_goal = pad if use_dxdy else 0.0
                dy_to_goal = pad if use_dxdy else 0.0

            f6, f7 = (cell_features_67.get(cell_num, (0.0, 0.0))
                      if not is_pad_cell else (pad, pad))

            vec = self._build_feature_vec(
                cell_num, x, y, status,
                dx_to_goal, dy_to_goal, f6, f7, pad
            )
            context_matrix[i] = torch.tensor(vec, dtype=torch.float32)

        foundroute_matrix = torch.full(
            (self.foundroute_matrix_length, self.feature_size), pad, dtype=torch.float32
        )
        for i, cell in enumerate(found_routes):
            if i >= self.foundroute_matrix_length:
                break
            cell_num = cell.get('Cell_Number', padding_value)
            x        = cell.get('x', padding_value)
            y        = cell.get('y', padding_value)
            status   = cell.get('State_Status', self.config["status_free"])

            is_pad_cell = (cell_num == padding_value or
                           x == padding_value or y == padding_value)
            if not is_pad_cell:
                dx_to_goal = float(cell.get("dx_to_goal", goal_x - x)) if use_dxdy else 0.0
                dy_to_goal = float(cell.get("dy_to_goal", goal_y - y)) if use_dxdy else 0.0
            else:
                dx_to_goal = pad if use_dxdy else 0.0
                dy_to_goal = pad if use_dxdy else 0.0

            f6, f7 = (cell_features_67.get(cell_num, (0.0, 0.0))
                      if not is_pad_cell else (pad, pad))

            vec = self._build_feature_vec(
                cell_num, x, y, status,
                dx_to_goal, dy_to_goal, f6, f7, pad
            )
            foundroute_matrix[i] = torch.tensor(vec, dtype=torch.float32)
        input_tensor = torch.cat([context_matrix, foundroute_matrix], dim=0)
        num_directions = int(self.config.get("num_unique_tokens", 4))
        masking_tensor = torch.full(
            (self.foundroute_matrix_length, num_directions), pad, dtype=torch.float32
        )
        for i, mask_entry in enumerate(masking_data):
            if i >= self.foundroute_matrix_length:
                break
            mask_vec = mask_entry.get("mask_vec", [padding_value] * num_directions)
            masking_tensor[i] = torch.tensor(mask_vec, dtype=torch.float32)

        target_tensor = torch.full(
            (self.target_matrix_length, num_directions), pad, dtype=torch.float32
        )
        for i, cell in enumerate(target):
            if i >= self.target_matrix_length:
                break
            target_vec = cell.get("target_vec", [padding_value] * num_directions)
            target_tensor[i] = torch.tensor(target_vec, dtype=torch.float32)

        padding_value_f = float(self.padding_value)

        mask_has_pad = masking_tensor.eq(padding_value_f).any(dim=-1)
        assert (mask_has_pad == masking_tensor.eq(padding_value_f).all(dim=-1)).all(), \
            "Invalid Masking row: mixed padding values."

        target_has_pad = target_tensor.eq(padding_value_f).any(dim=-1)
        assert (target_has_pad == target_tensor.eq(padding_value_f).all(dim=-1)).all(), \
            "Invalid Target row: mixed padding values."

        real_mask_rows = ~mask_has_pad
        if real_mask_rows.any():
            assert (masking_tensor[real_mask_rows].eq(1).any(dim=-1)).all(), \
                "Invalid Masking row: real step has no allowed actions."

        real_target_rows = ~target_has_pad
        if real_target_rows.any():
            assert (target_tensor[real_target_rows].sum(dim=-1).eq(1)).all(), \
                "Invalid Target row: target_vec must be one-hot."

        return input_tensor, masking_tensor, target_tensor, context, found_routes

def collate_fn(batch):
    input_tensors, masking_tensors, target_tensors, contexts, found_routes_list = zip(*batch)
    return (
        torch.stack(input_tensors),
        torch.stack(masking_tensors),
        torch.stack(target_tensors),
        contexts,
        found_routes_list
    )

class RAGatePtDataset(Dataset):
    def __init__(self, pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"RAGate .pt file not found: {pt_path}\n"
                f"Generate it with the A* dataset-generation repo and copy Datasets/RAGate/ here."
            )
        print(f"--> Loading RAGate dataset: {pt_path}")
        self._samples = torch.load(pt_path, weights_only=False)
        print(f"<-- Loaded {len(self._samples):,} RAGate samples from {pt_path}\n")
        for _i in _probe_indices(len(self._samples)):
            check_sample_against_config(self._samples[_i], pt_path)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        s = self._samples[idx]
        context     = raw_context(s)
        found_routes= raw_found_routes(s)
        label       = int(s.get("label", -1))
        return (
            s["input_tensor"],
            s["masking_tensor"],
            s["target_tensor"],
            label,
            context,
            found_routes,
        )

    def get_raw(self, idx):
        """Return the full raw sample dict — used by UNIFIED visualisation."""
        return self._samples[idx]

def ragate_collate_fn(batch):
    input_tensors, masking_tensors, target_tensors, labels, contexts, found_routes_list = zip(*batch)
    return (
        torch.stack(input_tensors),
        torch.stack(masking_tensors),
        torch.stack(target_tensors),
        list(labels),
        contexts,
        found_routes_list,
    )

def get_ragate_dataloader(pt_path, config, shuffle=False):
    dataset = RAGatePtDataset(pt_path)
    loader  = DataLoader(
        dataset,
        batch_size=config.get("ragate_embeddings_batch_size", 128),
        shuffle=shuffle,
        collate_fn=ragate_collate_fn,
        num_workers=0,           
        pin_memory=config.get("pin_memory", False),
    )
    return loader, dataset

def get_dataloader(file_path, config, pickle_key=None, base_dataset=None, shuffle=None, generator=None):
    from utils import seed_worker
    ds = base_dataset or PathfindingDataset(file_path, config, pickle_key)
    return DataLoader(
        ds,
        batch_size=config["batch"],
        shuffle=config.get("shuffle", False) if shuffle is None else bool(shuffle),
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
        generator=generator,
        worker_init_fn=seed_worker,
    )
