import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The script defines the basis function for tracing the squirrel.
# Therefore, a user can initially select the squirrel manually. 
# Subsequently, a buffer will be created around the squirrel. 
# This buffer will then be used as a mask to detect changes. 

# Further the user can select the entrance of the box end press enter, which 
# allows to measure the distance between the squirrel and the box (pixelwise)
# and to see if a squirrel has enteres the box

# The algorithm is very good at tracking the squirrel, but struggles to find it again after it enters the box.

# -----------------------------
# Config
# -----------------------------
VIDEO_PATH = "trimshort.mp4"  # <-- anpassen

WINDOW = "Squirrel ROI Tracker | click squirrel | then click entrance + Enter | q=quit | r=reset"

# Playback speed
SLOW_DELAY_MS = 250   # before selection
RUN_DELAY_MS  = 60    # tracking speed

# ROI / Buffer
ROI_RADIUS = 160
MIN_CHANGED_PIXELS = 300

# Change Detection
DIFF_THRESHOLD = 18
BLUR_K = 5

# Distance / logic
IN_BOX_DIST_PX = 35        # <= this distance => "squirrel in box"
OUT_BOX_DIST_PX = 60       # >= this distance => "squirrel out again" (hysteresis)
STABLE_FRAMES_IN = 5       # require N frames in a row to confirm IN
STABLE_FRAMES_OUT = 5      # require N frames in a row to confirm OUT

# Exports
OUT_CSV  = "distance_trace.csv"
OUT_PLOT = "distance_plot.png"


# -----------------------------
# State
# -----------------------------
state = {
    "phase": "WAIT_SQUIRREL",   # WAIT_SQUIRREL -> WAIT_ENTRANCE -> TRACKING
    "squirrel_center": None,    # (x,y)
    "entrance_point": None,     # (x,y)
    "prev_gray": None,
    "frame_idx": 0,

    # distance logging
    "dist_rows": [],

    # in-box logic
    "in_box": False,
    "in_counter": 0,
    "out_counter": 0,
    "timer_start_ms": None,
    "timer_rows": [],           # store (t_in, t_out, duration_s)
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_circular_mask(h, w, center, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = center
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask

# Compute change centroid in the ROI, return count of changed pixels, centroid, and binary diff mask
def compute_change_centroid(prev_gray, gray_now, mask):
    diff = cv2.absdiff(prev_gray, gray_now)
    diff_roi = cv2.bitwise_and(diff, diff, mask=mask)

    _, diff_bin = cv2.threshold(diff_roi, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    changed_count = int(cv2.countNonZero(diff_bin))
    if changed_count < MIN_CHANGED_PIXELS:
        return changed_count, None, diff_bin

    M = cv2.moments(diff_bin, binaryImage=True)
    if M["m00"] == 0:
        return changed_count, None, diff_bin

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return changed_count, (cx, cy), diff_bin

# Helper function to compute Euclidean distance between two points (or return None if either is None)
def distance_px(p1, p2):
    if p1 is None or p2 is None:
        return None
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.sqrt(dx * dx + dy * dy))


def draw_overlay(frame, roi_center, roi_radius, entrance_pt, diff_bin, text_lines):
    overlay = frame.copy()

    # ROI
    if roi_center is not None:
        cv2.circle(overlay, roi_center, roi_radius, (0, 0, 255), 2)
        cv2.circle(overlay, roi_center, 4, (0, 0, 255), -1)

    # Entrance
    if entrance_pt is not None:
        cv2.circle(overlay, entrance_pt, 6, (255, 255, 0), -1)
        cv2.putText(
            overlay, "entrance", (entrance_pt[0] + 8, entrance_pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

    # Show change pixels
    if diff_bin is not None:
        overlay[diff_bin > 0] = (0, 255, 0)

    # Text
    y = 40
    for line in text_lines:
        cv2.putText(
            overlay, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
        y += 35

    return overlay

# function to handle mouse clicks for selecting squirrel and entrance
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state["phase"] == "WAIT_SQUIRREL":
        state["squirrel_center"] = (x, y)
        state["phase"] = "WAIT_ENTRANCE"
        state["prev_gray"] = None  # will be set after entrance confirm
        print(f"[Squirrel selected] {state['squirrel_center']}")

    elif state["phase"] == "WAIT_ENTRANCE":
        state["entrance_point"] = (x, y)
        print(f"[Entrance selected] {state['entrance_point']} (press Enter to confirm)")

# function to compute distance and track squirrel (called in main loop)
def export_distance_plot(csv_path, plot_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[Plot] No data to plot.")
        return

    plt.figure()
    plt.plot(df["frame"], df["dist_px"])
    plt.xlabel("Frame")
    plt.ylabel("Distance (px)")
    plt.title("Squirrel-to-Entrance Distance (pixel)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

# --- Main Code ---
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        state["frame_idx"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # preprocess for change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

        key = 0

        # -------------------------
        # Phase 1: select squirrel
        # -------------------------
        if state["phase"] == "WAIT_SQUIRREL":
            overlay = draw_overlay(
                frame,
                roi_center=None,
                roi_radius=ROI_RADIUS,
                entrance_pt=None,
                diff_bin=None,
                text_lines=["Click squirrel to start"]
            )
            cv2.imshow(WINDOW, overlay)
            key = cv2.waitKey(SLOW_DELAY_MS) & 0xFF

        # -------------------------
        # Phase 2: select entrance + confirm
        # -------------------------
        elif state["phase"] == "WAIT_ENTRANCE":
            overlay = draw_overlay(
                frame,
                roi_center=state["squirrel_center"],
                roi_radius=ROI_RADIUS,
                entrance_pt=state["entrance_point"],
                diff_bin=None,
                text_lines=["Select entrance (click) ثم press Enter to confirm"]
            )
            cv2.imshow(WINDOW, overlay)

            # Pause video here until Enter (or reset/quit)
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):  # Enter
                if state["entrance_point"] is None:
                    print("[Warn] Please click entrance first.")
                else:
                    state["phase"] = "TRACKING"
                    state["prev_gray"] = gray.copy()  # initialize from current frame
                    state["in_box"] = False
                    state["in_counter"] = 0
                    state["out_counter"] = 0
                    state["timer_start_ms"] = None
                    print("[Tracking started]")
            # continue to controls section below (q/r)

        # -------------------------
        # Phase 3: tracking + distance + timer
        # -------------------------
        else:  # TRACKING
            roi_center = state["squirrel_center"]
            entrance_pt = state["entrance_point"]

            # ROI mask
            mask = make_circular_mask(h, w, roi_center, ROI_RADIUS)

            # change detection in ROI
            changed_count, new_center, diff_bin = compute_change_centroid(state["prev_gray"], gray, mask)

            # update ROI center if change exists
            if new_center is not None:
                nx = clamp(new_center[0], 0, w - 1)
                ny = clamp(new_center[1], 0, h - 1)
                state["squirrel_center"] = (nx, ny)

            # compute distance (px) between ROI center and entrance
            dist = distance_px(state["squirrel_center"], entrance_pt)

            # log
            if dist is not None:
                state["dist_rows"].append({
                    "frame": state["frame_idx"],
                    "time_s": state["frame_idx"] / fps,
                    "dist_px": dist,
                    "changed_px": changed_count,
                    "cx": state["squirrel_center"][0],
                    "cy": state["squirrel_center"][1],
                    "ex": entrance_pt[0],
                    "ey": entrance_pt[1],
                })

            # -----------------------------
            # In-box detection + timer logic (with hysteresis)
            # -----------------------------
            now_ms = cv2.getTickCount() * 1000.0 / cv2.getTickFrequency()

            if dist is not None and dist <= IN_BOX_DIST_PX:
                state["in_counter"] += 1
                state["out_counter"] = 0
            elif dist is not None and dist >= OUT_BOX_DIST_PX:
                state["out_counter"] += 1
                state["in_counter"] = 0
            else:
                # in the hysteresis band: don't flip, but also don't grow counters aggressively
                state["in_counter"] = max(0, state["in_counter"] - 1)
                state["out_counter"] = max(0, state["out_counter"] - 1)

            # confirm IN
            if (not state["in_box"]) and state["in_counter"] >= STABLE_FRAMES_IN:
                state["in_box"] = True
                state["timer_start_ms"] = now_ms
                print(f"[IN BOX] frame={state['frame_idx']} dist={dist:.1f}px")

            # confirm OUT
            if state["in_box"] and state["out_counter"] >= STABLE_FRAMES_OUT:
                state["in_box"] = False
                if state["timer_start_ms"] is not None:
                    duration_s = (now_ms - state["timer_start_ms"]) / 1000.0
                    state["timer_rows"].append({
                        "enter_frame": state["frame_idx"],
                        "duration_s": duration_s
                    })
                    print(f"[OUT BOX] frame={state['frame_idx']} duration={duration_s:.2f}s")
                state["timer_start_ms"] = None

            # build overlay text
            lines = []
            if dist is None:
                lines.append("dist: n/a")
            else:
                lines.append(f"dist_px: {dist:.1f} | changed_px: {changed_count}")
            if state["in_box"]:
                t = 0.0
                if state["timer_start_ms"] is not None:
                    t = (now_ms - state["timer_start_ms"]) / 1000.0
                lines.append(f"STATE: IN BOX | timer: {t:.2f}s")
            else:
                lines.append("STATE: OUTSIDE")

            overlay = draw_overlay(
                frame,
                roi_center=state["squirrel_center"],
                roi_radius=ROI_RADIUS,
                entrance_pt=entrance_pt,
                diff_bin=diff_bin,
                text_lines=lines
            )

            cv2.imshow(WINDOW, overlay)
            key = cv2.waitKey(RUN_DELAY_MS) & 0xFF

        # -------------------------
        # Controls
        # -------------------------
        if key == ord("q") or key == 27:
            break

        if key == ord("r"):
            print("[Reset]")
            state["phase"] = "WAIT_SQUIRREL"
            state["squirrel_center"] = None
            state["entrance_point"] = None
            state["prev_gray"] = None
            state["dist_rows"] = []
            state["timer_rows"] = []
            state["in_box"] = False
            state["in_counter"] = 0
            state["out_counter"] = 0
            state["timer_start_ms"] = None

    cap.release()
    cv2.destroyAllWindows()

    # -----------------------------
    # Export distance + plot
    # -----------------------------
    if state["dist_rows"]:
        df = pd.DataFrame(state["dist_rows"])
        df.to_csv(OUT_CSV, index=False)
        export_distance_plot(OUT_CSV, OUT_PLOT)
        print("Saved:", OUT_CSV)
        print("Saved:", OUT_PLOT)
    else:
        print("No distance data collected (no tracking run completed).")

    # Optional: export timer events
    if state["timer_rows"]:
        pd.DataFrame(state["timer_rows"]).to_csv("inbox_durations.csv", index=False)
        print("Saved: inbox_durations.csv")


if __name__ == "__main__":
    main()
