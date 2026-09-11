from collections import Counter, namedtuple


# ══════════════════════════════════════════════════════════════════════════════
# SHARED GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_neighbors(cell, grid_size):
    n_rows, n_cols = grid_size
    row = (cell - 1) // n_cols
    col = (cell - 1) % n_cols
    return [
        (row - 1, col),
        (row + 1, col),
        (row,     col + 1),
        (row,     col - 1),
    ]


def _cell_from_rc(r, c, n_cols):
    return r * n_cols + c + 1


def _manhattan(cell_a, cell_b, n_cols):
    ra, ca = (cell_a - 1) // n_cols, (cell_a - 1) % n_cols
    rb, cb = (cell_b - 1) // n_cols, (cell_b - 1) % n_cols
    return abs(ra - rb) + abs(ca - cb)


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — TOP-LEVEL CLASSIFIER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_failed_empty_path(pred_cells, **kw):
    return len(pred_cells) == 0


def check_max_steps_unreached(pred_cells, max_path_len, **kw):
    return len(pred_cells) >= max_path_len


def check_stuck_boundary_corner(boundary_count, **kw):
    """3+ sides are grid edges — agent cornered by the border."""
    return boundary_count >= 3


def check_stuck_full_obstacle(valid_moves, obstacle_count, **kw):
    """All reachable neighbours are obstacles — physically walled in."""
    return valid_moves == 0 and obstacle_count > 0


def check_failed_loops(pred_cells, **kw):
    """More than 30% of path cells are revisits — agent is circling."""
    if len(pred_cells) == len(set(pred_cells)):
        return False
    cell_counts   = Counter(pred_cells)
    revisit_count = sum(1 for count in cell_counts.values() if count > 1)
    return revisit_count > len(pred_cells) * 0.3


def check_failed_wrong_direction(pred_cells, goal_cell, grid_size, **kw):
    """Net motion increased Manhattan distance to goal."""
    if len(pred_cells) < 3:
        return False
    n_cols = grid_size[1]
    sample = [pred_cells[0], pred_cells[len(pred_cells) // 2], pred_cells[-1]]
    dists  = [_manhattan(c, goal_cell, n_cols) for c in sample]
    return dists[-1] > dists[0]


def check_stuck_revisit_3plus(neighbors, visited, state_status_map, grid_size, config, **kw):
    """3+ non-obstacle neighbours already visited — strict mask cornered agent."""
    n_rows, n_cols = grid_size
    blocked = 0
    for r, c in neighbors:
        if 0 <= r < n_rows and 0 <= c < n_cols:
            cell_num = _cell_from_rc(r, c, n_cols)
            if (cell_num in visited and
                    state_status_map.get(cell_num, 0) != config["status_obstacle"]):
                blocked += 1
    return blocked >= 3


def check_stuck_all_masked_top(term_reason, **kw):
    """Rollout stopped because all mask directions were zero."""
    return str(term_reason).strip().lower() in (
        "stuck_all_masked", "no_valid_actions", "masked_out"
    )


def check_unknown_stuck(**kw):
    """Catch-all — always matches. Must stay last."""
    return True


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

FAILURE_REGISTRY = [
    # name                  color           check fn                    active
    ("failed_empty_path",          "gray",         check_failed_empty_path,           True),
    ("max_steps_unreached",          "black",        check_max_steps_unreached,           True),
    ("stuck_boundary_corner",      "deepskyblue",  check_stuck_boundary_corner,       True),
    ("stuck_full_obstacle",      "purple",       check_stuck_full_obstacle,       True),
    ("failed_loops",               "cyan",         check_failed_loops,                True),
    ("failed_wrong_direction",     "magenta",      check_failed_wrong_direction,      True),
    ("stuck_revisit_3plus",       "gold",         check_stuck_revisit_3plus,        True),
    ("stuck_all_masked",        "orange",       check_stuck_all_masked_top,     True),
    ("unknown_stuck",       "brown",        check_unknown_stuck,        True),
]


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — STUCK_MASKED SUB-CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

MaskAnalysis = namedtuple("MaskAnalysis", [
    "last_cell",   # int — cell where the agent got stuck
    "n_boundary",  # int — directions blocked by the grid edge
    "n_obstacle",  # int — directions blocked by an obstacle neighbour
    "n_visited",   # int — directions blocked by an already-visited cell
    "n_free",      # int — genuinely free directions (0 for a stuck_masked failure)
    "sub_reason",  # str — assigned stuck_masked subtype
])


def _analyse_mask_state(last_cell, pred_cells, goal_cell, grid_size,
                        state_status_map, visited, config):
    """Count why each of the four neighbours of the stuck cell is blocked."""
    n_rows, n_cols = grid_size
    n_boundary = n_obstacle = n_visited = n_free = 0
    for r, c in _get_neighbors(last_cell, grid_size):
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            n_boundary += 1
            continue
        cell_num = _cell_from_rc(r, c, n_cols)
        if state_status_map.get(cell_num, 0) == config["status_obstacle"]:
            n_obstacle += 1
        elif cell_num in visited:
            n_visited += 1
        else:
            n_free += 1
    return dict(n_boundary=n_boundary, n_obstacle=n_obstacle,
                n_visited=n_visited, n_free=n_free)


# ── Sub-classifier functions (first match wins; n_free == 0 for stuck_masked) ──

def subcheck_boundary_corner(n_boundary, **kw):
    """Trapped at a grid corner: two or more sides are the grid edge."""
    return n_boundary >= 2


def subcheck_boundary_side(n_boundary, **kw):
    """Trapped on a grid edge (not a corner): exactly one side is the grid edge."""
    return n_boundary == 1


def subcheck_isolated(n_obstacle, **kw):
    """Sealed off by obstacles (interior cell): at least one obstacle neighbour."""
    return n_obstacle >= 1


def subcheck_self_trap(**kw):
    """Catch-all: an interior cell whose every remaining direction is an
    already-visited cell — the route enclosed itself."""
    return True


# ── Sub-classification registry (order = priority; first match wins) ──────────

SUB_CLASSIFIER_REGISTRY = [
    # name              color         check fn                   active
    ("boundary_corner", "navy",       subcheck_boundary_corner,  True),
    ("boundary_side",   "steelblue",  subcheck_boundary_side,    True),
    ("isolated",        "darkviolet", subcheck_isolated,         True),
    ("self_trap",       "darkorange", subcheck_self_trap,        True),
]


def classify_stuck_masked(last_cell, pred_cells, goal_cell, grid_size,
                          state_status_map, visited, config):
    """
    Run the stuck_masked sub-classifier and return a MaskAnalysis namedtuple
    with all diagnostic fields and the assigned sub_reason.

    Call this from route_evaluation.py for every route where
    term_reason == 'stuck_masked'.
    """
    fields = _analyse_mask_state(
        last_cell, pred_cells, goal_cell, grid_size,
        state_status_map, visited, config
    )
    sub_reason = "self_trap"
    for name, _color, check_fn, active in SUB_CLASSIFIER_REGISTRY:
        if not active:
            continue
        if check_fn(**fields):
            sub_reason = name
            break

    return MaskAnalysis(last_cell=last_cell, sub_reason=sub_reason, **fields)


def get_sub_classifier_names():
    """Return ordered list of active sub-classifier names."""
    return [name for name, _, _, active in SUB_CLASSIFIER_REGISTRY if active]


def get_sub_color_map():
    """Return {name: color} for all active sub-classifier types."""
    return {name: color for name, color, _, active in SUB_CLASSIFIER_REGISTRY if active}


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — CONVENIENCE HELPERS  (used by route_evaluation.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_all_failure_names():
    """Return ordered list of all active top-level failure names."""
    return [name for name, _, _, active in FAILURE_REGISTRY if active]


def get_color_map():
    """Return {name: color} for all active top-level failure types."""
    return {name: color for name, color, _, active in FAILURE_REGISTRY if active}


def classify(pred_cells, goal_cell, context, grid_size, state_status_map,
             visited, neighbors, valid_moves, obstacle_count, boundary_count,
             max_path_len, term_reason, config):
    """
    Run top-level classifiers in registry order.
    Returns the first matching failure name, or 'unknown_stuck'.
    term_reason is forwarded so check_stuck_all_masked_top can use it.
    """
    kwargs = dict(
        pred_cells       = pred_cells,
        goal_cell        = goal_cell,
        context          = context,
        grid_size        = grid_size,
        state_status_map = state_status_map,
        visited          = visited,
        neighbors        = neighbors,
        valid_moves      = valid_moves,
        obstacle_count   = obstacle_count,
        boundary_count   = boundary_count,
        max_path_len     = max_path_len,
        term_reason      = term_reason,
        config           = config,
    )
    for name, _color, check_fn, active in FAILURE_REGISTRY:
        if not active:
            continue
        if check_fn is None:
            continue
        if check_fn(**kwargs):
            return name
    return "unknown_stuck"
