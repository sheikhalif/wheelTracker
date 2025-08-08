import cv2
import numpy as np
import math
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# HSV range for green marker detection (adjust if needed)
green_lower = np.array([40, 60, 60])
green_upper = np.array([90, 255, 255])

# Angular label font
font = cv2.FONT_HERSHEY_SIMPLEX

# Read input video

cap = cv2.VideoCapture(0)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('annotated_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

initialized = False
init_time = None
label_A, label_B = None, None
angle_diff = None
avg_length = None
center_point = None
rotation_direction = None
start_A = 0
start_B = 0
recording = False
record_data = []
record_filename = f"angle_record_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
start_time = None

def angle_between(p1, p2):
    dx, dy = p2[0]-p1[0], p1[1]-p2[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    return angle

def midpoint(p1, p2):
    return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

def draw_angle_line(img, center, pt, color, label):
    cv2.line(img, center, pt, color, 2)
    angle = angle_between(center, pt)
    label_pos = (pt[0]+10, pt[1])
    cv2.putText(img, f'{label}: {angle:.1f}°', label_pos, font, 0.5, color, 1)

def interpolate_point(center_point, visible_pt, angle_offset_deg, rotation_direction):
    cx, cy = center_point
    base_angle = angle_between(center_point, visible_pt)

    # Determine direction of offset
    if rotation_direction == "CCW":
        use_angle = (base_angle + angle_offset_deg) % 360
    else:  # CW
        use_angle = (base_angle - angle_offset_deg) % 360

    # Convert to radians
    angle_rad = np.radians(use_angle)

    # Calculate interpolated point
    x = int(cx + avg_length * np.cos(angle_rad))
    y = int(cy - avg_length * np.sin(angle_rad))  # Subtract because y-axis is downward in image coords

    return (x, y)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for circular blobs of reasonable size
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30:
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

    state = "NOT INITIALIZED"
    if not initialized and len(centers) == 3:
        # Compute pairwise distances
        dists = []
        for i in range(3):
            for j in range(i+1, 3):
                d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                dists.append((d, i, j))
        dists.sort()
        # Shortest total distance point is center
        pairs = [dists[0][1], dists[0][2], dists[1][1], dists[1][2]]
        candidates = [x for x in pairs if pairs.count(x) > 1]
        center_idx = candidates[0]
        center_point = centers[center_idx]
        outer_points = [p for i, p in enumerate(centers) if i != center_idx]

        # Determine which is A and B by clockwise order
        angles = [angle_between(center_point, p) for p in outer_points]
        if angles[0] < angles[1]:
            label_A, label_B = outer_points[0], outer_points[1]
        else:
            label_A, label_B = outer_points[1], outer_points[0]
        start_A = label_A
        start_B = label_B

        # Set average length and angle diff
        lengths = [np.linalg.norm(np.array(center_point) - np.array(p)) for p in outer_points]
        avg_length = sum(lengths) / 2
        angle_diff = abs(angles[0] - angles[1])
        angle_diff = 360 - angle_diff if angle_diff > 180 else angle_diff

        initialized = True
        init_time = time.time()

    if len(centers) < 2:
        initialized = False
    if initialized:
        state = "INITIALIZED"
        cv2.putText(frame, state, (10, 25), font, 0.8, (0, 255, 0), 2)
        if recording:
            if start_time is None:
                start_time = time.time()
            current_time = time.time() - start_time
            angle_A = angle_between(center_point, label_A)
            record_data.append([current_time, angle_A])


        # Draw vertical blue line
        x = int(center_point[0])
        y = int(center_point[1] - avg_length)
        cv2.line(frame, center_point, (x, y), (255, 0, 0), 2)

        # Check visibility of A and B
        visible = { "A": False, "B": False }
        for pt in centers:
            if np.linalg.norm(np.array(pt) - np.array(label_A)) < 100:
                label_A = pt
                visible["A"] = True
                delta = (angle_between(center_point, label_A) - angle_between(center_point, start_A) + 360) % 360
                rotation_direction = "CCW" if 0 < delta < 180 else "CW"
            elif np.linalg.norm(np.array(pt) - np.array(label_B)) < 100:
                label_B = pt
                visible["B"] = True
                delta = (angle_between(center_point, label_B) - angle_between(center_point, start_B) + 360) % 360
                rotation_direction = "CCW" if 0 < delta < 180 else "CW"

        # A
        if visible["A"]:
            draw_angle_line(frame, center_point, label_A, (0, 255, 0), "A")
        else:
            interp_A = interpolate_point(center_point, label_B, angle_diff, "CW")
            draw_angle_line(frame, center_point, interp_A, (0, 255, 255), "A")
            label_A = interp_A

        # B
        if visible["B"]:
            draw_angle_line(frame, center_point, label_B, (0, 255, 0), "B")
        else:
            interp_B = interpolate_point(center_point, label_A, angle_diff, "CCW")
            draw_angle_line(frame, center_point, interp_B, (0, 255, 255), "B")
            label_B = interp_B

        start_A = label_A
        start_B = label_B
    if recording and len(record_data) > 2:
        # Plot the last N points
        times, angles = zip(*record_data[-100:])
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        ax.plot(times, angles, color='green')
        ax.set_title("Angle A vs Time")
        ax.set_xlabel("s")
        ax.set_ylabel("°")
        ax.set_ylim(0, 360)
        fig.tight_layout()
    
        # Render to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

    	# Resize and overlay plot on top-left
        h, w, _ = plot_img.shape
        plot_resized = cv2.resize(plot_img, (w//2, h//2))
        plot_resized = cv2.cvtColor(plot_resized, cv2.COLOR_RGBA2BGR)
        frame[10:10 + plot_resized.shape[0], 10:10 + plot_resized.shape[1]] = plot_resized
    #out.write(frame)
    cv2.imshow("Live Angle Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r') and initialized:
        recording = not recording
        if not recording and record_data:
            with open(record_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "angle_deg"])
                writer.writerows(record_data)
            print(f"[INFO] Saved angle data to {record_filename}")
            record_data = []
            start_time = None

cap.release()
out.release()
cv2.destroyAllWindows()
