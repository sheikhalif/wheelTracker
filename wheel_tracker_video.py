#!/usr/bin/env python3
"""
EvapoFlex wheel tracker + master-plot generator (unified).

Detects the green markers on the wheel, computes cumulative rotation using
the 180°-apart-perimeter constraint, and produces:
  • CSV of (time, raw angle, unwrapped angle, frame number)
  • Best-30s-RPM video clip
  • Single annotated master-plot PNG with summary stats

Usage examples:
    python3 wheel_tracker_video.py EED09_Test_1_Trimmed.mov
    python3 wheel_tracker_video.py EED09_Test_1_04.29.26.mov 5
    python3 wheel_tracker_video.py video.mov --design EED09 --test 1 \
                                              --date 2026-04-29 --skip 5
"""

import argparse
import csv
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Configuration
# =============================================================================
GREEN_LOWER = np.array([45, 60, 40])    # tightened from [28,30,30]
GREEN_UPPER = np.array([95, 255, 220])  # tightened from [100,255,255]

MIN_BLOB_AREA              = 20      # px^2, rejects tiny green noise
DETECTION_TIMEOUT_S        = 300.0   # 5 min  - exit if no green ever seen
CALIBRATION_TIMEOUT_S      = 600.0   # 10 min - exit if calibration never converges
CALIBRATION_MIN_DETECTIONS = 60      # min total detections before attempting fit
CALIBRATION_MIN_BIN_COUNT  = 10      # min hits in a candidate bin to consider it
CALIBRATION_MIN_FRACTION   = 0.50    # >=50% of other detections must lie on circle
CALIBRATION_MIN_SPREAD_DEG = 8.0     # supporters must span >= this much arc (degrees)
BEST_CLIP_DURATION_S       = 30.0    # length of exported best-RPM clip
ROI_MARGIN_PX              = 60      # padding around the wheel for tracking ROI
PERIMETER_TOLERANCE_FRAC   = 0.20    # marker counts as "perimeter" if |d-r|/r < 20%
BIN_SIZE_PX                = 5       # spatial bin size for finding the center marker
ROLLING_WINDOW_S           = 1.0     # window for rolling-RPM derivative on the plot

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# =============================================================================
# Boxed progress display
# =============================================================================
class StatusBox:
    """Fixed-height status box that updates in place via ANSI cursor moves.

    Falls back to plain throttled prints when stdout is not a TTY (e.g. when
    redirecting to a file).
    """
    WIDTH = 72  # outer width, fits in any 72-col+ terminal

    def __init__(self, title, info_pairs):
        self.title       = title
        self.info_pairs  = list(info_pairs)
        self.use_ansi    = sys.stdout.isatty()
        self.first_paint = True
        self._last_plain = 0.0

        # Number of dynamic lines + closing border line that must be redrawn.
        self.n_dyn_lines = 3
        self._closing    = self._border('╚', '╝', '═')

    # -- low-level box drawing -----------------------------------------------
    def _border(self, l, r, fill):
        return l + fill * (self.WIDTH - 2) + r

    def _line(self, content):
        inner = self.WIDTH - 2
        if len(content) > inner:
            content = content[: inner - 1] + "…"
        return "║" + content.ljust(inner) + "║"

    # -- public API ----------------------------------------------------------
    def render_initial(self):
        """Print the full box once."""
        print(self._border("╔", "╗", "═"))
        print(self._line(" " + self.title.center(self.WIDTH - 4) + " "))
        print(self._border("╠", "╣", "═"))
        for label, val in self.info_pairs:
            print(self._line(f"  {label:<10}{val}"))
        print(self._border("╟", "╢", "─"))
        for _ in range(self.n_dyn_lines):
            print(self._line(""))
        print(self._closing)
        sys.stdout.flush()

    def update(self, status, progress=None, detail=""):
        """Refresh the dynamic portion (status, progress bar, detail line)."""
        if not self.use_ansi:
            now = time.monotonic()
            if now - self._last_plain < 2.0:
                return
            self._last_plain = now
            pct = f"{progress*100:5.1f}%" if progress is not None else "  -- "
            print(f"  [{pct}] {status} | {detail}")
            return

        # Build the three dynamic lines
        lines = [self._line(f"  Status:    {status}")]
        if progress is not None:
            bar_w   = 30
            filled  = max(0, min(bar_w, int(round(bar_w * progress))))
            bar     = "█" * filled + "░" * (bar_w - filled)
            pct     = f"{progress*100:5.1f}%"
            lines.append(self._line(f"  Progress:  [{bar}] {pct}"))
        else:
            lines.append(self._line(f"  Progress:  [{'─'*30}]   --"))
        lines.append(self._line(f"  Detail:    {detail}"))

        # Move cursor up past the (n_dyn_lines + closing border) lines, rewrite
        sys.stdout.write(f"\033[{self.n_dyn_lines + 1}A")
        for line in lines:
            sys.stdout.write("\r" + line + "\033[K\n")
        sys.stdout.write("\r" + self._closing + "\033[K\n")
        sys.stdout.flush()

    def finalize(self, summary_pairs, header="RESULTS"):
        """Print a separate summary box below the live box."""
        print()
        print(self._border("╔", "╗", "═"))
        print(self._line(f"  {header}"))
        print(self._border("╟", "╢", "─"))
        for label, val in summary_pairs:
            print(self._line(f"  {label:<24}{val}"))
        print(self._border("╚", "╝", "═"))
        sys.stdout.flush()

    def warn(self, msg):
        """Print a warning beneath the box without disturbing it."""
        # Just write below the existing closing border
        print(f"  ⚠  {msg}")
        sys.stdout.flush()


# =============================================================================
# CV helpers
# =============================================================================
def angle_between(p1, p2):
    """Angle from p1 to p2 in degrees, [0, 360), with image y flipped to math y."""
    dx, dy = p2[0] - p1[0], p1[1] - p2[1]
    return np.degrees(np.arctan2(dy, dx)) % 360


def find_green_centers(image):
    """Detect green blobs above MIN_BLOB_AREA. Returns list of (x, y) centers."""
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_BLOB_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        out.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return out


def try_calibrate(detections, min_radius=30.0):
    """Find wheel center and radius from a flat list of (x, y) green detections.

    Strategy: bin detections; for each candidate bin with enough hits, score by
    the FRACTION of other detections lying within ±10% of the median distance
    (its "supporters"). The true wheel center has every perimeter detection at
    distance r, yielding fraction ≈ 1.0.

    Disambiguation note: in the early frames of a slowly-rotating wheel, the
    perimeter marker hasn't moved much yet, so a perimeter-cluster candidate
    ALSO scores fraction ≈ 1.0 (its only supporter is the stationary center
    marker, which sits at radius r in one direction). The two candidates are
    mathematically symmetric on bin-count and support-count alone, so we add a
    third measure: the ANGULAR SPREAD of the supporters as seen from the
    candidate. The true center sees perimeter detections at multiple angles as
    the wheel rotates; a perimeter cluster only ever sees the center marker in
    a single direction. Once the wheel has rotated a few degrees, the true
    center wins decisively. Calibration is deferred (None returned) until at
    least one candidate exceeds CALIBRATION_MIN_SPREAD_DEG.
    """
    if len(detections) < CALIBRATION_MIN_DETECTIONS:
        return None, None

    bins = defaultdict(list)
    for x, y in detections:
        bins[(x // BIN_SIZE_PX, y // BIN_SIZE_PX)].append((x, y))

    top = sorted(bins.items(), key=lambda kv: -len(kv[1]))[:5]
    if not top or len(top[0][1]) < CALIBRATION_MIN_BIN_COUNT:
        return None, None

    arr  = np.asarray(detections, dtype=float)
    best = None  # (fraction, spread_deg, support, center, radius)

    for _, members in top:
        if len(members) < CALIBRATION_MIN_BIN_COUNT:
            continue
        cand_x = float(np.mean([m[0] for m in members]))
        cand_y = float(np.mean([m[1] for m in members]))
        d = np.sqrt((arr[:, 0] - cand_x) ** 2 + (arr[:, 1] - cand_y) ** 2)
        sel = d > 20.0
        d_other = d[sel]
        if d_other.size < 10:
            continue
        med = float(np.median(d_other))
        if med < min_radius:
            continue
        tol      = med * 0.10
        circle   = np.abs(d_other - med) < tol
        support  = int(np.sum(circle))
        fraction = support / d_other.size
        if support < 10:
            continue

        # Angular spread of supporters as seen from the candidate
        sup_x = arr[sel, 0][circle]
        sup_y = arr[sel, 1][circle]
        ang   = np.arctan2(-(sup_y - cand_y), sup_x - cand_x)  # math y up
        s     = np.sort(ang)
        gaps  = np.diff(np.concatenate([s, s[:1] + 2 * np.pi]))
        spread_deg = float(np.degrees(2 * np.pi - np.max(gaps)))

        cand_score = (fraction, spread_deg, support)
        if best is None or cand_score > (best[0], best[1], best[2]):
            best = (fraction, spread_deg, support,
                    (int(round(cand_x)), int(round(cand_y))), med)

    if (best is None
            or best[0] < CALIBRATION_MIN_FRACTION
            or best[1] < CALIBRATION_MIN_SPREAD_DEG):
        return None, None
    return best[3], best[4]


def update_track(centers, t, fnum, calib_center, calib_radius, state):
    """Append a record using the 180°-apart perimeter constraint.

    The angle of any visible perimeter marker is reduced mod 180°. Because A
    and B are diametrically opposite, A's angle and B's angle differ by exactly
    180°, so they collapse to the same mod-180 value. The unwrapped accumulator
    ends up in real degrees of cumulative wheel rotation.

    state: dict with keys 'prev_mod180' and 'unwrapped'. Mutated in place.
    """
    cx, cy   = calib_center
    tol      = max(40.0, PERIMETER_TOLERANCE_FRAC * calib_radius)
    best_pt  = None
    best_err = float("inf")
    for c in centers:
        d = math.hypot(c[0] - cx, c[1] - cy)
        err = abs(d - calib_radius)
        if err < tol and err < best_err:
            best_err = err
            best_pt  = c
    if best_pt is None:
        return None

    raw_angle = angle_between(calib_center, best_pt)
    mod180    = raw_angle % 180.0

    if state["prev_mod180"] is not None:
        diff = mod180 - state["prev_mod180"]
        if   diff >  90: diff -= 180
        elif diff < -90: diff += 180
        state["unwrapped"] += diff
    state["prev_mod180"] = mod180
    return [t, raw_angle, state["unwrapped"], fnum]


# =============================================================================
# Filename / metadata parsing
# =============================================================================
def parse_filename_metadata(path):
    """Best-effort extraction of design / test / date from a video filename."""
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    out = {"design": None, "test": None, "date_iso": None}

    m = re.search(r"(EED\d+)", stem, re.IGNORECASE)
    if m:
        out["design"] = m.group(1).upper()

    m = re.search(r"Test[\s_]?(\d+)", stem, re.IGNORECASE)
    if m:
        out["test"] = int(m.group(1))

    # Try ISO date YYYY-MM-DD first
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", stem)
    if m:
        out["date_iso"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        # Fall back to MM.DD.YY style
        m = re.search(r"(\d{2})\.(\d{2})\.(\d{2})", stem)
        if m:
            mm, dd, yy = m.group(1), m.group(2), m.group(3)
            yyyy = f"20{yy}" if int(yy) < 70 else f"19{yy}"
            out["date_iso"] = f"{yyyy}-{mm}-{dd}"
    return out


def pretty_date(iso_str):
    """'2026-04-29' -> 'Apr 29, 2026'."""
    dt = datetime.strptime(iso_str, "%Y-%m-%d")
    try:
        return dt.strftime("%b %-d, %Y")     # POSIX
    except ValueError:
        return dt.strftime("%b %#d, %Y")     # Windows fallback


def fmt_runtime(seconds):
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# =============================================================================
# Plotting (single PNG, RPM curve from rolling derivative of unwrapped angle)
# =============================================================================
def rolling_rpm(time_s, unwrap, window_s=ROLLING_WINDOW_S):
    """Rolling RPM via centered finite difference over `window_s`.

    For each sample i, finds the points within ±window_s/2 around time_s[i]
    and computes RPM = (Δangle / Δtime) / 6. When the centered window is too
    narrow to enclose another sample (sparse data, or large gaps between
    records), falls back to the immediate prev/next neighbors so the curve
    stays continuous instead of going all-NaN.
    """
    n = len(time_s)
    if n < 2:
        return np.full(n, np.nan)
    half = window_s / 2.0

    lo = np.searchsorted(time_s, time_s - half, side="left")
    hi = np.searchsorted(time_s, time_s + half, side="right") - 1
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(hi, 0, n - 1)

    # Where the centered window encloses no second sample, expand to immediate
    # neighbors. This guarantees rpm is non-NaN whenever n >= 2.
    too_narrow = hi <= lo
    if too_narrow.any():
        i_arr = np.arange(n)
        lo[too_narrow] = np.maximum(0,     i_arr[too_narrow] - 1)
        hi[too_narrow] = np.minimum(n - 1, i_arr[too_narrow] + 1)

    rpm   = np.full(n, np.nan)
    valid = hi > lo
    if valid.any():
        dt = time_s[hi[valid]] - time_s[lo[valid]]
        da = unwrap[hi[valid]] - unwrap[lo[valid]]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dt > 0, (da / dt) / 6.0, np.nan)
        rpm[valid] = r
    return rpm


def consecutive_flat_mask(values, diff_threshold=0.005, min_consecutive=3):
    """Mark stall points: |Δvalue| < threshold for ≥ min_consecutive in a row.
    Operates directly on whatever 1-D array is passed (use unwrapped angle so
    no wrap correction is needed)."""
    d = np.abs(np.diff(values))
    flat_step = np.concatenate(([False], d < diff_threshold))
    out = np.zeros_like(flat_step, dtype=bool)
    run = 0
    for i, ok in enumerate(flat_step):
        if ok:
            run += 1
            if run >= min_consecutive:
                out[i - min_consecutive + 1: i + 1] = True
        else:
            run = 0
    return out


def make_master_plot(records, design, test_n, date_pretty, png_path,
                     best_window=None):
    """Render the single annotated master plot to png_path.
    Returns a metrics dict.
    """
    arr       = np.asarray(records, dtype=float)
    time_s    = arr[:, 0]
    raw_deg   = arr[:, 1]   # noqa: F841 (kept for completeness)
    unwrap    = arr[:, 2]

    # Metrics
    avg_dps    = (unwrap[-1] - unwrap[0]) / (time_s[-1] - time_s[0]) \
                 if time_s[-1] > time_s[0] else 0.0
    avg_rpm    = (avg_dps / 360.0) * 60.0
    stall_mask = consecutive_flat_mask(unwrap, diff_threshold=0.05,
                                        min_consecutive=3)
    diffs      = np.diff(unwrap)
    rebound_mask = np.concatenate(([False], diffs < -0.001))
    continuity_pct = (1 - float(np.mean(stall_mask)))    * 100.0
    motion_pct     = (1 - float(np.mean(rebound_mask)))  * 100.0
    efficiency_pct = float(np.mean((~stall_mask) & (~rebound_mask))) * 100.0
    runtime_str    = fmt_runtime(time_s[-1] - time_s[0])

    # Cycle averages (per master_plot.py logic, applied to unwrapped data)
    rpm_curve = rolling_rpm(time_s, unwrap, window_s=ROLLING_WINDOW_S)

    # ---- Plot ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(14, 6.5))

    # Zero reference for positive vs. negative rotation
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5, zorder=1)

    ax.plot(time_s, rpm_curve, color="steelblue", linewidth=1.3,
            label=f"RPM ({ROLLING_WINDOW_S:g}s rolling)", zorder=2)

    if best_window is not None:
        ax.axvspan(best_window["start_t"], best_window["end_t"],
                   color="gold", alpha=0.15,
                   label=f"Best 30s: {best_window['rpm']:.3f} RPM")

    ax.set_title(f"{design} Test {test_n} — {date_pretty}", fontsize=13)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Explicit limits so a sparse / mostly-NaN curve still spans the full
    # time range and the y-axis covers all real values plus the zero line.
    ax.set_xlim(float(time_s[0]), float(time_s[-1]))

    finite_rpm = rpm_curve[np.isfinite(rpm_curve)]
    if finite_rpm.size > 0:
        y_min_d = float(finite_rpm.min())
        y_max_d = float(finite_rpm.max())
    else:
        y_min_d, y_max_d = -0.1, 0.1
    # Always include zero (axhline reference) inside the plot
    y_min_d = min(y_min_d, 0.0)
    y_max_d = max(y_max_d, 0.0)
    y_range = max(y_max_d - y_min_d, 0.1)
    # Pad: 10% below, 22% above (room for the summary box in the upper-right)
    ax.set_ylim(y_min_d - 0.10 * y_range, y_max_d + 0.22 * y_range)

    # Summary text box (upper-right)
    summary = (
        f"Total Run Time     {runtime_str}\n"
        f"Average RPM        {avg_rpm:.3f}\n"
        f"Motion Score       {motion_pct:5.2f}%\n"
        f"Continuity Score   {continuity_pct:5.2f}%\n"
        f"Overall Efficiency {efficiency_pct:5.2f}%"
    )
    ax.text(0.99, 0.97, summary,
            transform=ax.transAxes,
            ha="right", va="top",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="lightgray", linewidth=0.8))

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return {
        "runtime":        runtime_str,
        "avg_rpm":        avg_rpm,
        "motion_pct":     motion_pct,
        "continuity_pct": continuity_pct,
        "efficiency_pct": efficiency_pct,
    }


# =============================================================================
# Best-window search
# =============================================================================
def find_best_window(records, window_s):
    if len(records) < 2:
        return None
    arr       = np.asarray(records, dtype=float)
    times     = arr[:, 0]
    unwrapped = arr[:, 2]
    frames    = arr[:, 3].astype(int)
    n = len(times)
    best, j = None, 0
    for i in range(n):
        if j <= i:
            j = i + 1
        while j < n and times[j] - times[i] < window_s:
            j += 1
        if j >= n:
            break
        dt = times[j] - times[i]
        if dt < window_s * 0.95 or dt > window_s * 1.5:
            continue
        rpm = abs((unwrapped[j] - unwrapped[i]) / dt) / 6.0
        if best is None or rpm > best["rpm"]:
            best = {
                "start_t":     float(times[i]),
                "end_t":       float(times[j]),
                "start_frame": int(frames[i]),
                "end_frame":   int(frames[j]),
                "rpm":         float(rpm),
                "n_samples":   int(j - i + 1),
            }
    return best


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="EvapoFlex wheel tracker + master-plot generator.")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("skip_pos", nargs="?", type=int, default=None,
                    help="Optional positional frame skip (legacy).")
    ap.add_argument("--skip", type=int, default=None,
                    help="Process every Nth frame (default 1)")
    ap.add_argument("--design", type=str, default=None,
                    help="Design code (e.g. EED09). Auto-extracted from "
                         "filename if absent.")
    ap.add_argument("--test", type=int, default=None,
                    help="Test number. Auto-extracted from filename if absent.")
    ap.add_argument("--date", type=str, default=None,
                    help="Video date as YYYY-MM-DD. Auto-extracted from "
                         "filename, then file mtime, if absent.")
    ap.add_argument("--out-dir", type=str, default=".",
                    help="Output directory (default '.').")
    args = ap.parse_args()

    # Resolve metadata: CLI > filename > file-mtime > default
    auto = parse_filename_metadata(args.video)
    design  = args.design or auto["design"] or "EED??"
    test_n  = args.test   if args.test is not None else (auto["test"] or 1)
    date_iso = args.date or auto["date_iso"]
    if date_iso is None and os.path.exists(args.video):
        mtime = os.path.getmtime(args.video)
        date_iso = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    if date_iso is None:
        date_iso = datetime.now().strftime("%Y-%m-%d")
    date_pretty = pretty_date(date_iso)

    skip = args.skip if args.skip is not None else (args.skip_pos or 1)
    if skip < 1:
        skip = 1

    wallclock_start = time.monotonic()

    # Helper: cumulative average RPM from the first to the most recent record.
    # records is the live record_data list; we just need its first and last
    # entries (each is [t_s, angle_deg, unwrapped_deg, frame_num]).
    def _cumulative_rpm(records):
        if len(records) < 2:
            return None
        dt = records[-1][0] - records[0][0]
        da = records[-1][2] - records[0][2]
        if dt <= 0:
            return None
        return (da / dt) / 6.0

    def _fmt_rpm(v):
        return f"{v:+.3f}" if v is not None else "  ---"

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / fps if fps > 0 else 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix  = f"{design}_Test_{test_n}_{date_iso}"
    csv_path    = os.path.join(args.out_dir, f"{out_prefix}_angle.csv")
    png_path    = os.path.join(args.out_dir, f"{out_prefix}_master_plot.png")
    clip_path   = os.path.join(args.out_dir, f"{out_prefix}_best_clip.mp4")

    # Status box
    title = f"EvapoFlex Wheel Tracker  ·  {design} Test {test_n}"
    box = StatusBox(title, [
        ("Video",  os.path.basename(args.video)),
        ("Size",   f"{frame_w}×{frame_h} @ {fps:.1f} fps"),
        ("Length", f"{total_frames} frames ({duration_s/60:.1f} min)"),
        ("Skip",   f"every {skip} frame{'s' if skip>1 else ''}  ·  "
                    f"date {date_pretty}"),
    ])
    box.render_initial()

    # =========================================================================
    # Phase 1: calibration (full-frame detection)
    # =========================================================================
    calib_buffer       = []
    flat_detections    = []
    first_green_seen_t = None
    calibrated         = False
    calib_center       = None
    calib_radius       = None
    last_attempt_at_n  = 0
    last_ui_t          = 0.0

    frame_num = 0

    while cap.isOpened() and not calibrated:
        # Use grab() (no decode) for skipped frames; retrieve() only when
        # this is a frame we'll actually process. Skips run ~3-4× faster.
        if not cap.grab():
            break
        frame_num += 1
        if skip > 1 and frame_num % skip != 0:
            continue
        ret, frame = cap.retrieve()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        centers = find_green_centers(frame)

        if centers and first_green_seen_t is None:
            first_green_seen_t = t

        # 5-min "no green ever" timeout
        if first_green_seen_t is None and t >= DETECTION_TIMEOUT_S:
            box.warn(f"No green markers detected within "
                     f"{DETECTION_TIMEOUT_S/60:.0f} min. Adjust GREEN_LOWER/UPPER.")
            cap.release()
            sys.exit(2)

        if centers:
            calib_buffer.append((centers, t, frame_num))
            flat_detections.extend(centers)

        # Try fitting periodically
        if len(flat_detections) - last_attempt_at_n >= 50:
            last_attempt_at_n = len(flat_detections)
            c, r = try_calibrate(flat_detections)
            if c is not None:
                calib_center, calib_radius, calibrated = c, r, True

        # 10-min calibration timeout
        if (not calibrated) and t >= CALIBRATION_TIMEOUT_S:
            c, r = try_calibrate(flat_detections)
            if c is not None:
                calib_center, calib_radius, calibrated = c, r, True
            else:
                box.warn(f"Calibration failed within "
                         f"{CALIBRATION_TIMEOUT_S/60:.0f} min.")
                cap.release()
                sys.exit(3)

        # Throttle UI updates to ~5/s
        now = time.monotonic()
        if now - last_ui_t > 0.2:
            last_ui_t = now
            prog = frame_num / max(total_frames, 1)
            green_state = "yes" if first_green_seen_t else "no"
            elapsed = fmt_runtime(now - wallclock_start)
            box.update(
                "Phase 1 / 3  ·  calibrating wheel center",
                progress=prog,
                detail=f"t={t:6.1f}s  green={green_state}  "
                       f"detections={len(flat_detections)}  elapsed={elapsed}",
            )

    if not calibrated:
        # End of video without success - one final attempt
        c, r = try_calibrate(flat_detections) if flat_detections else (None, None)
        if c is not None:
            calib_center, calib_radius, calibrated = c, r, True
        else:
            box.warn("Calibration failed - no usable wheel circle found.")
            cap.release()
            sys.exit(4)

    box.update("Phase 1 / 3  ·  calibration complete", progress=None,
               detail=f"center={calib_center}  radius={calib_radius:.1f}px  "
                      f"({len(flat_detections)} samples)")

    # ROI
    roi_x = max(0, int(calib_center[0] - calib_radius - ROI_MARGIN_PX))
    roi_y = max(0, int(calib_center[1] - calib_radius - ROI_MARGIN_PX))
    roi_w = min(frame_w - roi_x, int(2 * calib_radius + 2 * ROI_MARGIN_PX))
    roi_h = min(frame_h - roi_y, int(2 * calib_radius + 2 * ROI_MARGIN_PX))
    speedup = (frame_w * frame_h) / max(1, roi_w * roi_h)

    # =========================================================================
    # Phase 2a: track buffered calibration data
    # =========================================================================
    record_data = []
    state       = {"prev_mod180": None, "unwrapped": 0.0}

    for centers, t, fnum in calib_buffer:
        rec = update_track(centers, t, fnum, calib_center, calib_radius, state)
        if rec is not None:
            record_data.append(rec)

    n_buffered = len(record_data)
    calib_buffer.clear()
    flat_detections.clear()

    box.update("Phase 2 / 3  ·  tracking (ROI)", progress=None,
               detail=f"buffered records={n_buffered}  "
                      f"ROI speedup ≈ {speedup:.1f}×")

    # =========================================================================
    # Phase 2b: continue forward, ROI-only detection
    # =========================================================================
    last_ui_t = 0.0
    while cap.isOpened():
        if not cap.grab():
            break
        frame_num += 1
        if skip > 1 and frame_num % skip != 0:
            continue
        ret, frame = cap.retrieve()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        sub         = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        sub_centers = find_green_centers(sub)
        centers     = [(c[0] + roi_x, c[1] + roi_y) for c in sub_centers]

        rec = update_track(centers, t, frame_num, calib_center,
                           calib_radius, state)
        if rec is not None:
            record_data.append(rec)

        now = time.monotonic()
        if now - last_ui_t > 0.2:
            last_ui_t = now
            prog = frame_num / max(total_frames, 1)
            rpm_now = _cumulative_rpm(record_data)
            elapsed = fmt_runtime(now - wallclock_start)
            box.update(
                "Phase 2 / 3  ·  tracking forward (ROI)",
                progress=prog,
                detail=f"t={t:6.1f}s  rec={len(record_data)}  "
                       f"avg RPM={_fmt_rpm(rpm_now)}  elapsed={elapsed}",
            )

    cap.release()

    # =========================================================================
    # Save CSV
    # =========================================================================
    if record_data:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "angle_deg", "unwrapped_angle_deg",
                        "frame_num"])
            w.writerows(record_data)

    # =========================================================================
    # Best 30s clip
    # =========================================================================
    best = find_best_window(record_data, BEST_CLIP_DURATION_S)
    clip_written = False
    if best is not None:
        box.update("Phase 3 / 3  ·  exporting best clip & plot",
                   progress=None,
                   detail=f"best 30s ≈ {best['rpm']:.3f} RPM "
                          f"(t={best['start_t']:.1f}–{best['end_t']:.1f}s)")
        cap2 = cv2.VideoCapture(args.video)
        if cap2.isOpened():
            cap2.set(cv2.CAP_PROP_POS_FRAMES, best["start_frame"])

            # Always export a VERTICAL clip. If the source is landscape
            # (W > H), rotate every frame 90° clockwise; if it is already
            # portrait (W <= H), pass it through unchanged.
            if frame_w > frame_h:
                out_w, out_h = frame_h, frame_w
                rotate_clip  = True
            else:
                out_w, out_h = frame_w, frame_h
                rotate_clip  = False

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(clip_path, fourcc, fps, (out_w, out_h))
            if not out.isOpened():
                clip_path = clip_path.replace(".mp4", ".avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(clip_path, fourcc, fps, (out_w, out_h))
            cur = best["start_frame"]
            while cur <= best["end_frame"]:
                ok, frm = cap2.read()
                if not ok:
                    break
                if rotate_clip:
                    frm = cv2.rotate(frm, cv2.ROTATE_90_CLOCKWISE)
                out.write(frm)
                cur += 1
            out.release()
            cap2.release()
            clip_written = True

    # =========================================================================
    # Master plot
    # =========================================================================
    metrics = None
    if record_data:
        box.update("Phase 3 / 3  ·  rendering master plot",
                   progress=None, detail=os.path.basename(png_path))
        metrics = make_master_plot(record_data, design, test_n, date_pretty,
                                   png_path, best_window=best)

    # =========================================================================
    # Final summary box
    # =========================================================================
    elapsed_total = fmt_runtime(time.monotonic() - wallclock_start)

    box.update("Done", progress=1.0,
               detail=(f"{len(record_data)} records  ·  "
                       f"avg {metrics['avg_rpm']:.3f} RPM  ·  "
                       f"elapsed {elapsed_total}")
                       if metrics
                       else f"{len(record_data)} records  ·  elapsed {elapsed_total}")

    summary_pairs = [
        ("Records",            f"{len(record_data)}"),
        ("Processing Time",    elapsed_total),
    ]
    if metrics is not None:
        summary_pairs += [
            ("Total Run Time",     metrics["runtime"]),
            ("Average RPM",        f"{metrics['avg_rpm']:.3f}"),
            ("Motion Score",       f"{metrics['motion_pct']:.2f}%"),
            ("Continuity Score",   f"{metrics['continuity_pct']:.2f}%"),
            ("Overall Efficiency", f"{metrics['efficiency_pct']:.2f}%"),
        ]
    if best is not None:
        summary_pairs.append(
            ("Best 30s Window RPM", f"{best['rpm']:.3f}"))
    summary_pairs += [
        ("CSV",                os.path.basename(csv_path) if record_data else "—"),
        ("Master Plot",        os.path.basename(png_path) if metrics else "—"),
        ("Best Clip",          os.path.basename(clip_path) if clip_written else "—"),
    ]
    box.finalize(summary_pairs, header=f"RESULTS  ·  {design} Test {test_n}  ·  "
                                       f"{date_pretty}")


if __name__ == "__main__":
    main()