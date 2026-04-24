from __future__ import annotations

import csv
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyboed_mpl")
import matplotlib
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE_ORDER = ["conservative", "incremental"]


def build_setups(base_setup, sweep):
    """Build the Cartesian product of the requested setup values."""
    sweep_keys = list(sweep.keys())
    if not sweep_keys:
        return [
            {
                "setup_id": "setup_000",
                "setup_label": "base",
                "setup": dict(base_setup),
                "varying_keys": [],
            }
        ]

    value_lists = []
    for key in sweep_keys:
        values = sweep[key]
        if isinstance(values, np.ndarray):
            values = values.tolist()
        elif not isinstance(values, (list, tuple)):
            values = [values]
        values = list(values)
        if not values:
            raise ValueError(f"Sweep for '{key}' must contain at least one value.")
        value_lists.append(values)

    setups = []
    seen = set()
    for combo in itertools.product(*value_lists):
        signature = tuple(_format_scalar(value) for value in combo)
        if signature in seen:
            continue
        seen.add(signature)

        setup = dict(base_setup)
        setup.update(dict(zip(sweep_keys, combo)))
        setups.append(
            {
                "setup_id": f"setup_{len(setups):03d}",
                "setup_label": format_setup_label(setup, sweep_keys),
                "setup": setup,
                "varying_keys": sweep_keys,
            }
        )
    return setups


def format_setup_label(setup, keys):
    if not keys:
        return "base"
    return " | ".join(f"{key}={_format_scalar(setup[key])}" for key in keys)


def make_seed(base_seed, setup_index, repeat_index):
    return int(base_seed) + 1000 * int(setup_index) + int(repeat_index)


def sanitize_sensor_budgets(sensor_budgets, n_grid):
    budgets = sorted(
        {
            int(budget)
            for budget in sensor_budgets
            if int(budget) > 0 and int(budget) <= int(n_grid)
        }
    )
    if not budgets:
        raise ValueError(
            f"No valid sensor budgets remain after filtering with N={int(n_grid)}."
        )
    return budgets


def initialize_results_file(output_dir, parameter_names, metadata_fields=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_path / "raw_results.csv"
    metadata_fields = list(metadata_fields or [])

    fieldnames = (
        ["mode", "phase", "setup_id", "setup_label", "repeat", "seed", "budget"]
        + list(parameter_names)
        + metadata_fields
        + ["lb", "ub", "gap", "elapsed_phase_s"]
    )

    with raw_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    return output_path, raw_csv_path, fieldnames


def write_run_config(output_dir, payload):
    config_path = Path(output_dir) / "run_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default, sort_keys=True)


def append_rows(csv_path, fieldnames, rows):
    rows = list(rows)
    if not rows:
        return

    with Path(csv_path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def refresh_outputs(output_dir, parameter_names):
    output_path = Path(output_dir)
    raw_csv_path = output_path / "raw_results.csv"
    rows = load_rows(raw_csv_path)
    if not rows:
        return

    summary_rows = build_summary_rows(rows, parameter_names)
    write_summary_csv(output_path / "summary.csv", parameter_names, summary_rows)

    phases = [phase for phase in PHASE_ORDER if any(r["phase"] == phase for r in rows)]
    for phase in phases:
        plot_bounds_boxplots(
            rows=rows,
            phase=phase,
            output_path=output_path / f"{phase}_bounds_boxplots.png",
        )
        plot_runtime_boxplots(
            rows=rows,
            phase=phase,
            output_path=output_path / f"{phase}_runtime_boxplots.png",
        )


def load_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["repeat"] = int(row["repeat"])
            row["seed"] = int(row["seed"])
            row["budget"] = int(row["budget"])
            row["lb"] = float(row["lb"])
            row["ub"] = float(row["ub"])
            row["gap"] = float(row["gap"])
            row["elapsed_phase_s"] = float(row["elapsed_phase_s"])
            rows.append(row)
    return rows


def build_summary_rows(rows, parameter_names):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["setup_id"], row["budget"])].append(row)

    summary_rows = []
    for phase, setup_id, budget in sorted(
        grouped,
        key=lambda key: (_phase_rank(key[0]), key[1], key[2]),
    ):
        bucket = grouped[(phase, setup_id, budget)]
        sample = bucket[0]
        lb_values = np.array([row["lb"] for row in bucket], dtype=float)
        ub_values = np.array([row["ub"] for row in bucket], dtype=float)
        gap_values = np.array([row["gap"] for row in bucket], dtype=float)
        elapsed_values = np.array([row["elapsed_phase_s"] for row in bucket], dtype=float)

        summary_row = {
            "mode": sample["mode"],
            "phase": phase,
            "setup_id": setup_id,
            "setup_label": sample["setup_label"],
            "budget": budget,
            "n_repeats": len(bucket),
            "lb_median": float(np.median(lb_values)),
            "lb_q1": float(np.quantile(lb_values, 0.25)),
            "lb_q3": float(np.quantile(lb_values, 0.75)),
            "ub_median": float(np.median(ub_values)),
            "ub_q1": float(np.quantile(ub_values, 0.25)),
            "ub_q3": float(np.quantile(ub_values, 0.75)),
            "gap_median": float(np.median(gap_values)),
            "gap_q1": float(np.quantile(gap_values, 0.25)),
            "gap_q3": float(np.quantile(gap_values, 0.75)),
            "elapsed_phase_median_s": float(np.median(elapsed_values)),
        }
        for name in parameter_names:
            summary_row[name] = sample.get(name, "")
        summary_rows.append(summary_row)
    return summary_rows


def write_summary_csv(summary_path, parameter_names, summary_rows):
    fieldnames = (
        ["mode", "phase", "setup_id", "setup_label", "budget", "n_repeats"]
        + list(parameter_names)
        + [
            "lb_median",
            "lb_q1",
            "lb_q3",
            "ub_median",
            "ub_q1",
            "ub_q3",
            "gap_median",
            "gap_q1",
            "gap_q3",
            "elapsed_phase_median_s",
        ]
    )
    with Path(summary_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_bounds_boxplots(rows, phase, output_path):
    phase_rows = [row for row in rows if row["phase"] == phase]
    if not phase_rows:
        return

    setup_ids = _ordered_setup_ids(phase_rows)
    n_rows, n_cols = _subplot_layout(len(setup_ids))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.7 * n_cols, 4.2 * n_rows),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    values = [row["lb"] for row in phase_rows] + [row["ub"] for row in phase_rows]
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    margin = 0.05 * (y_max - y_min + 1e-12)

    for ax, setup_id in zip(axes, setup_ids):
        setup_rows = [row for row in phase_rows if row["setup_id"] == setup_id]
        budgets = sorted({row["budget"] for row in setup_rows})
        positions = np.asarray(budgets, dtype=float)
        min_step = float(np.min(np.diff(positions))) if len(positions) > 1 else 1.0
        box_width = 0.32 * min_step
        offset = 0.20 * min_step

        lb_data = [
            [row["lb"] for row in setup_rows if row["budget"] == budget]
            for budget in budgets
        ]
        ub_data = [
            [row["ub"] for row in setup_rows if row["budget"] == budget]
            for budget in budgets
        ]

        lb_box = ax.boxplot(
            lb_data,
            positions=positions - offset,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops=dict(color="tab:blue"),
            whiskerprops=dict(color="tab:blue"),
            capprops=dict(color="tab:blue"),
            medianprops=dict(color="tab:blue", linewidth=1.6),
        )
        ub_box = ax.boxplot(
            ub_data,
            positions=positions + offset,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops=dict(color="tab:orange"),
            whiskerprops=dict(color="tab:orange"),
            capprops=dict(color="tab:orange"),
            medianprops=dict(color="tab:orange", linewidth=1.6),
        )
        for box in lb_box["boxes"]:
            box.set_facecolor("tab:blue")
            box.set_alpha(0.22)
        for box in ub_box["boxes"]:
            box.set_facecolor("tab:orange")
            box.set_alpha(0.22)

        lb_median = [float(np.median(values)) for values in lb_data]
        ub_median = [float(np.median(values)) for values in ub_data]
        ax.plot(positions, lb_median, color="tab:blue", marker="o", linewidth=1.8)
        ax.plot(positions, ub_median, color="tab:orange", marker="s", linewidth=1.8)

        ax.set_title(setup_rows[0]["setup_label"].replace(" | ", "\n"), fontsize=10)
        ax.set_xlabel("Number of sensors")
        ax.set_xticks(budgets)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Information gain")
    legend_handles = [
        Line2D([0], [0], color="tab:blue", marker="o", linewidth=1.8, label="LB median"),
        Line2D(
            [0],
            [0],
            color="tab:orange",
            marker="s",
            linewidth=1.8,
            label="UB median",
        ),
    ]
    axes[0].legend(handles=legend_handles, title="Boxes: repeats", fontsize=9)

    for ax in axes[len(setup_ids) :]:
        ax.axis("off")

    fig.suptitle(f"{phase.capitalize()} bounds across repeated runs", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_runtime_boxplots(rows, phase, output_path):
    phase_rows = [row for row in rows if row["phase"] == phase]
    if not phase_rows:
        return

    elapsed_by_setup = defaultdict(list)
    labels = {}
    seen = set()
    for row in phase_rows:
        key = (row["setup_id"], row["repeat"])
        if key in seen:
            continue
        seen.add(key)
        elapsed_by_setup[row["setup_id"]].append(row["elapsed_phase_s"])
        labels[row["setup_id"]] = row["setup_label"]

    setup_ids = _ordered_setup_ids(phase_rows)
    data = [elapsed_by_setup[setup_id] for setup_id in setup_ids]

    fig, ax = plt.subplots(1, 1, figsize=(max(6.5, 1.8 * len(setup_ids)), 4.6))
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("tab:green")
        patch.set_alpha(0.22)
        patch.set_edgecolor("tab:green")

    medians = [float(np.median(values)) for values in data]
    ax.plot(
        np.arange(1, len(setup_ids) + 1),
        medians,
        color="tab:green",
        marker="o",
        linewidth=1.8,
    )

    ax.set_xticks(np.arange(1, len(setup_ids) + 1))
    ax.set_xticklabels(
        [labels[setup_id].replace(" | ", "\n") for setup_id in setup_ids],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title(f"{phase.capitalize()} runtime across repeated runs")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def eig_BS(Sigma_signal, Sigma_Y_given_theta, W):
    eye = np.eye(Sigma_Y_given_theta.shape[0])
    return 0.5 * (
        _log_ratio(Sigma_signal, Sigma_Y_given_theta, W)
        - _log_ratio(Sigma_signal, Sigma_Y_given_theta, eye)
    )


def eig_BI(Sigma_Y, Sigma_noise, W):
    eye = np.eye(Sigma_noise.shape[0])
    return 0.5 * (
        _log_ratio(Sigma_Y, Sigma_noise, W) - _log_ratio(Sigma_Y, Sigma_noise, eye)
    )


def greedy_maximize_lb(Sigma_Y, Sigma_noise, n_sensors):
    n_grid = Sigma_Y.shape[0]
    selected = []
    remaining = list(range(n_grid))

    for _ in range(n_sensors):
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            candidate = selected + [idx]
            index = np.ix_(candidate, candidate)
            sign_y, logdet_y = np.linalg.slogdet(Sigma_Y[index])
            sign_n, logdet_n = np.linalg.slogdet(Sigma_noise[index])
            if sign_y <= 0 or sign_n <= 0:
                raise ValueError(f"Sub-matrix is not SPD for S={candidate}")
            score = 0.5 * (logdet_y - logdet_n)
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return np.asarray(selected, dtype=int)


def incremental_bounds(
    Sigma_signal,
    Sigma_Y_theta,
    Sigma_Y,
    Sigma_noise,
    n_sensors,
):
    n_grid = Sigma_Y.shape[0]
    candidates = [np.eye(n_grid)[i] for i in range(n_grid)]
    selected = []
    remaining = list(range(n_grid))
    scores_inf = []
    scores_sup = []

    for _ in range(n_sensors):
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            trial = selected + [idx]
            W = np.column_stack([candidates[i] for i in trial])
            score = _signed_log_ratio(Sigma_signal, Sigma_Y_theta, W)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        W = np.column_stack([candidates[i] for i in selected])
        scores_inf.append(_log_ratio(Sigma_signal, Sigma_Y_theta, W))
        scores_sup.append(_log_ratio(Sigma_Y, Sigma_noise, W))

    return {
        "indices": np.asarray(selected, dtype=int),
        "scores_inf": scores_inf,
        "scores_sup": scores_sup,
        "EIG_lower_bound": scores_inf[-1] if scores_inf else 0.0,
        "EIG_upper_bound": scores_sup[-1] if scores_sup else None,
    }


def _phase_rank(phase):
    return PHASE_ORDER.index(phase) if phase in PHASE_ORDER else len(PHASE_ORDER)


def _subplot_layout(n_plots):
    if n_plots <= 3:
        return 1, max(1, n_plots)
    if n_plots <= 6:
        return 2, math.ceil(n_plots / 2)
    return 3, math.ceil(n_plots / 3)


def _ordered_setup_ids(rows):
    ordered = []
    seen = set()
    for row in rows:
        setup_id = row["setup_id"]
        if setup_id in seen:
            continue
        seen.add(setup_id)
        ordered.append(setup_id)
    return ordered


def _format_scalar(value):
    if isinstance(value, (np.floating, float)):
        return f"{float(value):g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _json_default(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _log_ratio(A, B, M):
    _, logdet_a = np.linalg.slogdet(M.T @ A @ M)
    _, logdet_b = np.linalg.slogdet(M.T @ B @ M)
    return 0.5 * (logdet_a - logdet_b)


def _signed_log_ratio(A, B, M):
    sign_a, logdet_a = np.linalg.slogdet(M.T @ A @ M)
    sign_b, logdet_b = np.linalg.slogdet(M.T @ B @ M)
    if sign_a <= 0 or sign_b <= 0:
        return -np.inf
    return 0.5 * (logdet_a - logdet_b)
