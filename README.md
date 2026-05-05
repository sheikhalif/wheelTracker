# **DEPENDENCIES**
You will need to install the following libraries to run the scripts in this repo:

`pip install pandas numpy matplotlib opencv-python`

# UPDATE 5/5/26 (wheel_tracker_video.py)
## Detection
- HSV range tightened to `[45,60,40]–[95,255,220]` (was `[35,40,40]–[95,255,255]`); rejects frame plastic, shelf bracket, and reflections
- `GaussianBlur` removed (no longer needed, saves ~2 ms/frame)
- `MIN_BLOB_AREA` lowered to 20 px²

## Calibration
- New two-phase calibration; no longer requires all 3 markers visible at once
- Center identified by circle-support fraction with an 8° angular-spread tiebreaker (resolves the early-rotation symmetry between true center and perimeter cluster)
- Buffered detections tracked retroactively so CSV starts at `t = 0`
- Exit codes: `2` = no green in 5 min, `3` = no calibration in 10 min, `4` = video ended unsolved

## Tracking
- Uses 180°-apart perimeter constraint: angle of any visible perimeter marker, mod 180°, is the same for A and B
- Eliminates the rebound oscillation that was caused by A/B identity confusion
- Post-calibration detection runs on a square ROI (~50% of frame, ~2× speedup)

## Plot
- `master_plot.py` integrated; single PNG output `{design}_Test_{N}_{date}_master_plot.png`
- Y-axis is rolling RPM (1s centered window over unwrapped angle); zero line; explicit `xlim`/`ylim` from data
- Gold band marks best 30s window
- Summary box (run time, avg RPM, motion/continuity/efficiency) anchored top-right
- Removed: per-cycle dashed lines, per-revolution RPM labels, stall/rebound dots

## Best clip
- Exports `{design}_Test_{N}_{date}_best_clip.mp4` (the highest-RPM 30s window)
- Auto-rotated to portrait when source is landscape
- AVI/XVID fallback if `mp4v` unavailable

## Terminal UI
- Replaced line-by-line prints with fixed-height ANSI status box (Status / Progress / Detail rewritten in place)
- Falls back to throttled text on non-TTY stdout
- Live detail shows: video time, record count, cumulative avg RPM, wallclock elapsed
- Final RESULTS box: records, processing time, run time, avg RPM, motion/continuity/efficiency, best-30s RPM, output filenames

## Performance
- `cap.read()` → `cap.grab()` + conditional `cap.retrieve()`: skipped frames advance without decoding
- Net: 53-minute video at `--skip 5` ≈ 22 min → ~11 min

## CLI
- New flags: `--skip`, `--design`, `--test`, `--date`, `--out-dir`
- Auto-extracts design/test/date from filenames matching `EED##`, `Test_#`, `MM.DD.YY`, `YYYY-MM-DD`
- Legacy positional `[skip]` still accepted

## CSV
- Schema unchanged: `time_s, angle_deg, unwrapped_angle_deg, frame_num`
- Note: `t = 0` is now start-of-video, not start-of-calibration

# **WHEEL TRACKER**
### **Purpose:**

The Wheel Tracker tool helps collect quantitative data on the spin cycle of Evaporation engines using a camera and computer vision. Please note that this tool assumes you are using a Mac and have an iPhone connected. Support for other cameras can be added later if needed.

### **Use Guide:**

On startup, the program will be on standby until it detects three green points: one to denote the center and two on opposite sides of the wheel. The program does not require a fixed angle spacing between the non-center points but they should be placed at least 90 degrees apart to ensure at least one point is visible on the wheel at all times. When all three points are detected, lines from the center to the outer points will be drawn and it will print "INITIALIZED" on the top left.
<p align="center">
  <img 
       src="https://github.com/user-attachments/assets/dd3c4a37-818e-4f56-b875-45f4c2e99075"
       alt="image"
       style="width: 40%; border: 1px solid #ddd; border-radius: 12px; padding: 6px; box-shadow: 2px 2px 12px rgba(0,0,0,0.1);" 
  >
</p>

Once the program is initialized, simply press "R" to start recording angle values into a CSV. A graph will be shown on the top left to keep track of your measurements:
<p align="center">
  <img 
       src="https://github.com/user-attachments/assets/f06b23a8-cf82-4ccd-ba64-ab934749797f"
       alt="image"
       style="width: 40%; border: 1px solid #ddd; border-radius: 12px; padding: 6px; box-shadow: 2px 2px 12px rgba(0,0,0,0.1);" 
  >
</p>

### **Output:**

Once you have collected your desired data, press "R" again to stop recording and press Esc to exit the script. A CSV with all your data will be created and you can refer to the terminal to verify the file name.


# **MASTER PLOT**
### **Purpose:**

The Master Plot tool is a one-step solution to analyzing raw CSV data to collect information about the wheel's performance and score it on various metrics. 

### **Use Guide:**

The script requires input arguments to execute in the format below:

`python3 master_plot.py --csv your_csv_file.csv --design design_code --test test_number`

The command requires the path to your raw data CSV file, the design code of the wheel, and the test iteration number. For example:

`python3 master_plot.py --csv angle_record_2025-07-15_19-41-27.csv --design EED03 --test 1`

### **Output:**

The script will output two .svg files: a plot showing individual cycle RPMs, stall points, and rebound points, along with a summary report containing experiment details and performance scores for different metrics.

<table>
  <tr>
    <td valign="middle" style="padding-right:12px;">
      <img
        src="https://github.com/user-attachments/assets/59377a8a-a84e-4b37-aaa0-7afda5b4d721"
        alt="EED03_Test_1_Master_Graph"
        height="400"
        style="border:1px solid #ddd;border-radius:8px;padding:6px;box-shadow:2px 2px 12px rgba(0,0,0,0.1);"
      >
    </td>
    <td valign="middle">
      <img
        src="https://github.com/user-attachments/assets/fe85b05d-391d-4503-9bd2-e05917ee7670"
        alt="EED03_Test_1_Summary_Report"
        height="400"
        style="border:1px solid #ddd;border-radius:8px;padding:12px;box-shadow:2px 2px 12px rgba(0,0,0,0.1);"
      >
    </td>
  </tr>
</table>

**Motion Score:** Percentage of time that the wheel is moving

**Continuity Score:** Percentage of time that the wheel is not rebounding

**Overall Efficiency:** Percentage of time that the wheel is moving continuously in the right direction





