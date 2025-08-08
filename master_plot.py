#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Master Plot generator

Run example:
  python master_plot.py --csv angle_record_2025-08-07_20-00-31.csv --design EED04 --test 1

Outputs:
  EED04_Test_1_Master_Graph.svg
  EED04_Test_1_Summary_Report.svg

Workflow (as agreed):
  • Round angle_deg to 0.1° for stall detection
  • Handle 360→0 wrap for velocity calcs
  • Graph:
      - Raw data (steelblue)
      - RPM average lines per cycle (gray dashed), labeled (shifted 50s left)
      - Stall points (orange) using <0.005°/step for ≥3 consecutive points
      - Rebound points (red) where angle decreases
  • Metrics:
      - Total Run Time: Xh Ym Zs
      - Average RPM: computed from raw deltas (wrap-aware)
      - Motion Score: % time not reversing
      - Continuity Score: % time not stalled
      - Overall Efficiency: % time both not reversing and not stalled
  • Summary SVG “card” with compact layout, no top padding above title
"""

import os
import re
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------- helpers ----------------------------

def parse_date_from_filename(path: str) -> str:
    """
    Extract YYYY-MM-DD from filenames like 'angle_record_2025-08-07_20-00-31.csv'
    and return 'Mon D, YYYY' (e.g., 'Aug 7, 2025').
    """
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', os.path.basename(path))
    if not m:
        raise ValueError("Could not find YYYY-MM-DD in filename.")
    dt = datetime.strptime(m.group(0), "%Y-%m-%d")
    # %−d is not portable; use platform-specific flag for no-leading-zero day
    try:
        return dt.strftime("%b %-d, %Y")  # POSIX
    except ValueError:
        return dt.strftime("%b %#d, %Y")  # Windows


def format_runtime(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def consecutive_flat_mask(angle_deg: np.ndarray,
                          diff_threshold: float = 0.005,
                          min_consecutive: int = 3) -> np.ndarray:
    """
    Mark 'stall' points: |Δangle| < threshold for at least `min_consecutive` in a row.
    Expects angles already rounded to 0.1°.
    """
    d = np.abs(np.diff(angle_deg))
    d = np.where(d > 180, 360 - d, d)  # wrap-aware
    flat_step = np.concatenate(([False], d < diff_threshold))

    out = np.zeros_like(flat_step, dtype=bool)
    run = 0
    for i, ok in enumerate(flat_step):
        if ok:
            run += 1
            if run >= min_consecutive:
                out[i - min_consecutive + 1:i + 1] = True
        else:
            run = 0
    return out


def rpm_from_raw(time_s: np.ndarray, angle_deg: np.ndarray) -> float:
    """Average RPM computed directly from raw (wrap-aware) deltas."""
    d = np.diff(angle_deg)
    d = np.where(d < -180, d + 360, d)
    d = np.where(d > 180, d - 360, d)
    dt = np.diff(time_s)
    avg_dps = np.sum(d) / np.sum(dt)
    return (avg_dps / 360.0) * 60.0


def cycle_average_lines(time_s: np.ndarray, angle_deg: np.ndarray):
    """
    Build cycle avg lines using unwrapped angles (360-crossing detected by floor division).
    Returns list of tuples: (mask, avg_line, start_idx, end_idx, rpm, ang_unwrap)
    """
    ang_unwrap = np.unwrap(np.deg2rad(angle_deg)) * 180 / np.pi
    bounds = [0]
    for i in range(1, len(ang_unwrap)):
        if int(ang_unwrap[i] // 360) != int(ang_unwrap[i - 1] // 360):
            bounds.append(i)
    bounds.append(len(ang_unwrap) - 1)

    lines = []
    for s_idx, e_idx in zip(bounds[:-1], bounds[1:]):
        if e_idx <= s_idx:
            continue
        total_angle = ang_unwrap[e_idx] - ang_unwrap[s_idx]
        total_time = time_s[e_idx] - time_s[s_idx]
        if total_time <= 0:
            continue
        slope_dps = total_angle / total_time
        rpm = (slope_dps / 360.0) * 60.0
        start_time = time_s[s_idx]
        start_angle = angle_deg[s_idx]
        avg_line = start_angle + slope_dps * (time_s - start_time)
        mask = (time_s >= start_time) & (time_s <= time_s[e_idx])
        lines.append((mask, avg_line, s_idx, e_idx, rpm, ang_unwrap))
    return lines


# ---------------------------- generators ----------------------------

def make_master_graph_svg(csv_path: str, design_code: str, out_prefix: str):
    """Create the combined annotated graph SVG and return (out_path, metrics dict)."""
    date_str = parse_date_from_filename(csv_path)

    df = pd.read_csv(csv_path)
    time = df["time_s"].to_numpy()
    angle_raw = df["angle_deg"].to_numpy()

    # Rounded copy for stall detection
    angle = angle_raw.round(1)

    # Components
    avg_lines = cycle_average_lines(time, angle)
    avg_rpm = rpm_from_raw(time, angle)

    # Masks
    stall_mask = consecutive_flat_mask(angle, diff_threshold=0.005, min_consecutive=3)
    decreasing_mask = np.concatenate(([False], np.diff(angle) < 0))  # motion direction

    # Scores
    continuity_score = (1 - np.mean(stall_mask)) * 100.0
    motion_score = (1 - np.mean(decreasing_mask)) * 100.0
    efficiency_score = np.mean((~stall_mask) & (~decreasing_mask)) * 100.0

    # Plot style
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10
    })

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, angle, color="steelblue", linewidth=1.2, label="Angle")

    for (mask, avg_line, s_idx, e_idx, rpm, ang_unwrap) in avg_lines:
        ax.plot(time[mask], avg_line[mask], linestyle="--", color="gray", linewidth=1)
        mid_t = (time[s_idx] + time[e_idx]) / 2.0
        start_t = time[s_idx]
        start_a = angle[s_idx]
        slope_dps = (ang_unwrap[e_idx] - ang_unwrap[s_idx]) / (time[e_idx] - start_t)
        mid_a = start_a + slope_dps * (mid_t - start_t)
        ax.text(mid_t - 50, mid_a, f"{rpm:.3f} RPM", fontsize=8, va="bottom", ha="center", color="black")

    ax.scatter(time[stall_mask], angle[stall_mask], color="darkorange", s=10, label="Stall Points")
    ax.scatter(time[decreasing_mask], angle[decreasing_mask], color="crimson", s=10, label="Rebound Points")

    ax.set_title(f"{design_code} — {date_str}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    out_graph = f"{out_prefix}_Master_Graph.svg"
    fig.savefig(out_graph, format="svg")
    plt.close(fig)

    # Metrics for summary
    total_runtime = format_runtime(time[-1] - time[0])
    metrics = {
        "date_str": date_str,
        "avg_rpm": avg_rpm,
        "motion_score": motion_score,
        "continuity_score": continuity_score,
        "efficiency_score": efficiency_score,
        "runtime": total_runtime,
        "design_code": design_code
    }
    return out_graph, metrics


def make_summary_svg(metrics: dict, out_prefix: str):
    """Create the compact summary SVG card."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(4.2, 1.65))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # No top padding title + rule
    title_y = 1.03
    ax.text(0.5, title_y, "Summary Report", fontsize=12, ha="center", va="top", weight="bold")
    ax.hlines(title_y - 0.03, 0, 1, color="black", linewidth=0.5)

    items = [
        ("Total Run Time", metrics["runtime"]),
        ("Average RPM", f"{metrics['avg_rpm']:.3f}"),
        ("Motion Score", f"{metrics['motion_score']:.2f}%"),
        ("Continuity Score", f"{metrics['continuity_score']:.2f}%"),
        ("Overall Efficiency", f"{metrics['efficiency_score']:.2f}%"),
    ]

    line_y = 0.985
    spacing = 0.045
    for label, value in items:
        ax.text(0.03, line_y, label, ha="left", va="top", fontsize=10, family="monospace")
        ax.text(0.62, line_y, value, ha="left", va="top", fontsize=10, family="monospace")
        line_y -= spacing

    ax.text(0.5, line_y, f"{metrics['design_code']} - {metrics['date_str']}",
            ha="center", va="top", fontsize=10)

    out_summary = f"{out_prefix}_Summary_Report.svg"
    fig.savefig(out_summary, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_summary


# ---------------------------- CLI ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Master Plot from raw wheel test data")
    parser.add_argument("--csv", required=True, help="Path to raw CSV file")
    parser.add_argument("--design", required=True, help="Design code, e.g., EED04")
    parser.add_argument("--test", required=True, type=int, help="Test number, e.g., 1")
    args = parser.parse_args()

    out_prefix = f"{args.design}_Test_{args.test}"

    graph_svg, metrics = make_master_graph_svg(args.csv, args.design, out_prefix)
    summary_svg = make_summary_svg(metrics, out_prefix)

    print(f"Saved:\n  {graph_svg}\n  {summary_svg}")


if __name__ == "__main__":
    main()

