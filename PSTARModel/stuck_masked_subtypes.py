from collections import namedtuple

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

def subcheck_boundary_corner(n_boundary, **kw):
    return n_boundary >= 2


def subcheck_boundary_side(n_boundary, **kw):
    return n_boundary == 1


def subcheck_isolated(n_obstacle, **kw):
    return n_obstacle >= 1


def subcheck_self_trap(**kw):
    return True

SUB_CLASSIFIER_REGISTRY = [
    # name              color         check fn                   active
    ("boundary_corner", "navy",       subcheck_boundary_corner,  True),
    ("boundary_side",   "steelblue",  subcheck_boundary_side,    True),
    ("isolated",        "darkviolet", subcheck_isolated,         True),
    ("self_trap",       "darkorange", subcheck_self_trap,        True),
]

def classify_stuck_masked(last_cell, pred_cells, goal_cell, grid_size,
                          state_status_map, visited, config):

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
    return [name for name, _, _, active in SUB_CLASSIFIER_REGISTRY if active]


def get_sub_color_map():
    return {name: color for name, color, _, active in SUB_CLASSIFIER_REGISTRY if active}

ROUTE_TYPE_SUCCESS = "reached_goal"
ROUTE_TYPES = (ROUTE_TYPE_SUCCESS,) + tuple(name for name, _, _, active in SUB_CLASSIFIER_REGISTRY if active)
ROUTE_TYPE_COLORS = {ROUTE_TYPE_SUCCESS: "green", **{n: c for n, c, _, a in SUB_CLASSIFIER_REGISTRY if a}}


def get_route_type_names():
    return list(ROUTE_TYPES)


def get_route_type_color_map():
    return dict(ROUTE_TYPE_COLORS)


def classify_route(pred_cells, goal_cell, grid_size, state_status_map, config):
    if pred_cells and goal_cell is not None and int(pred_cells[-1]) == int(goal_cell):
        return ROUTE_TYPE_SUCCESS, None
    if not pred_cells:
        return "self_trap", None
    analysis = classify_stuck_masked(
        last_cell=int(pred_cells[-1]), pred_cells=pred_cells, goal_cell=goal_cell,
        grid_size=grid_size, state_status_map=state_status_map,
        visited=set(int(c) for c in pred_cells), config=config)
        
    return analysis.sub_reason, analysis
