import csv
import json
import math
import os
from config import config

def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (None, None)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100.0 * max(0.0, centre - half), 3),
            round(100.0 * min(1.0, centre + half), 3))

def parse_template_groups(spec=None):
    spec = config.get("template_groups", "") if spec is None else spec
    groups = []
    for i, part in enumerate(p.strip() for p in str(spec).split(",") if p.strip()):
        lo, hi = (int(x) for x in part.split("-"))
        groups.append((f"G{i + 1}", {str(t) for t in range(lo, hi + 1)}))
    return groups

def group_of(template_id, groups):
    if template_id is None:
        return None
    for name, ids in groups:
        if str(template_id) in ids:
            return name
    return None

def _rate_block(rows):
    n = len(rows)
    succ = [r for r in rows if r.get("success")]
    k_s = len(succ)
    k_opt = sum(1 for r in succ if r.get("exact_optimal"))
    excess = [r["excess_steps"] for r in succ if r.get("excess_steps") is not None]
    ratio = [r["step_ratio"] for r in succ if r.get("step_ratio") is not None]
    return {
        "n": n,
        "success": k_s,
        "success_rate_pct": round(100.0 * k_s / n, 3) if n else None,
        "success_rate_ci95_pct": wilson_ci(k_s, n),
        "exact_optimal": k_opt,
        "exact_optimal_rate_pct": round(100.0 * k_opt / n, 3) if n else None,
        "exact_optimal_rate_ci95_pct": wilson_ci(k_opt, n),
        "exact_optimal_given_success_pct": round(100.0 * k_opt / k_s, 3) if k_s else None,
        "mean_excess_steps_success": round(sum(excess) / len(excess), 4) if excess else None,
        "max_excess_steps_success": max(excess) if excess else None,
        "mean_step_ratio_success": round(sum(ratio) / len(ratio), 4) if ratio else None,
    }

def _count_by(rows, key):
    out = {}
    for r in rows:
        v = r.get(key)
        if v not in (None, "", "success"):
            out[str(v)] = out.get(str(v), 0) + 1
    return out

def _route_type_counts(rows):
    try:
        from stuck_masked_subtypes import get_route_type_names
        names = get_route_type_names()
    except ImportError:
        names = []
    if not any(r.get("route_type") for r in rows):
        return {}                     # e.g. unified-framework records
    out = {n: 0 for n in names}
    for r in rows:
        t = r.get("route_type")
        if t:
            out[t] = out.get(t, 0) + 1
    return out


def write_reports(records, out_dir, prefix, extra_summary=None):
    os.makedirs(out_dir, exist_ok=True)
    groups = parse_template_groups()
    for r in records:
        r.setdefault("template_group", group_of(r.get("Template_ID"), groups))
    base_cols = ["dataset_index", "Template_ID", "template_group", "Open_Percent",
                 "Direction", "Pair_ID", "Split", "true_len", "pred_len", "success",
                 "exact_optimal", "excess_steps", "step_ratio", "termination",
                 "route_type", "failure_class", "stuck_subtype"]
    extra_cols = sorted({k for r in records for k in r} - set(base_cols))
    cols = base_cols + extra_cols
    sample_path = os.path.join(out_dir, f"{prefix}_per_sample.csv")
    with open(sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    by_t = {}
    for r in records:
        by_t.setdefault(str(r.get("Template_ID")), []).append(r)

    def _tkey(t):
        return (0, int(t)) if t.isdigit() else (1, t)

    tmpl_path = os.path.join(out_dir, f"{prefix}_per_template.csv")
    t_cols = ["Template_ID", "template_group", "n", "success", "success_rate_pct",
              "exact_optimal", "exact_optimal_rate_pct", "mean_excess_steps_success",
              "mean_step_ratio_success", "route_types"]
    with open(tmpl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=t_cols, extrasaction="ignore")
        w.writeheader()
        for t in sorted(by_t, key=_tkey):
            rows = by_t[t]
            blk = _rate_block(rows)
            blk.update({
                "Template_ID": t,
                "template_group": group_of(t, groups),
                "route_types": json.dumps(_route_type_counts(rows)),
            })
            w.writerow(blk)
    summary = {
        "grid_size": list(config["grid_size"]),
        "foundroute_matrix_length": int(config["foundroute_matrix_length"]),
        "overall": _rate_block(records),
        "route_types": _route_type_counts(records),
        "failures_by_route_type": _count_by([r for r in records if not r.get("success")],
                                            "route_type"),
        "by_group": {g: _rate_block([r for r in records if r.get("template_group") == g])
                     for g, _ in groups},
        "by_open_percent": {
            str(op): _rate_block([r for r in records if r.get("Open_Percent") == op])
            for op in sorted({r.get("Open_Percent") for r in records
                              if r.get("Open_Percent") is not None})},
        "num_templates": len(by_t),
        "files": {"per_sample": sample_path, "per_template": tmpl_path},
    }
    if extra_summary:
        summary.update(extra_summary)
    summary_path = os.path.join(out_dir, f"{prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    o = summary["overall"]
    print(f"--> [{prefix}] n={o['n']:,}  success={o['success_rate_pct']}% "
          f"CI95={o['success_rate_ci95_pct']}  exact-optimal={o['exact_optimal_rate_pct']}% "
          f"CI95={o['exact_optimal_rate_ci95_pct']}  mean excess={o['mean_excess_steps_success']}")
    print(f"--> [{prefix}] per-sample   : {sample_path}")
    print(f"--> [{prefix}] per-template : {tmpl_path}")
    print(f"--> [{prefix}] summary      : {summary_path}")
    return summary
