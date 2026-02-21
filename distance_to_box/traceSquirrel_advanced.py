# Squirrel Entrance Distance Tracker
# Tracks the distance from a manually selected squirrel to a manually selected entrance point.
# Logs distance over time and detects when the squirrel enters the box (based on distance threshold).

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
VIDEO_PATH = "trimshortEntrance.mp4"  # <-- adjust

WINDOW = "Squirrel ROI Tracker | click squirrel | then click entrance + Enter | q=quit | r=reset"

# Playback speed
SLOW_DELAY_MS = 250
RUN_DELAY_MS  = 10

# ROI / Buffer
ROI_RADIUS = 180
MIN_CHANGED_PIXELS = 200

# Change Detection
DIFF_THRESHOLD = 18
BLUR_K = 5

# Distance / logic
IN_BOX_ENTER_DIST_PX = 45      # <-- NEW: enter earlier (tune this)
IN_BOX_ZERO_DIST_PX  = 20      # if distance <= this => dist displayed/logged as 0

STABLE_FRAMES_IN = 5           # still require stability to confirm IN

# --- Reacquire (Mode 2) ---
REACQUIRE_RADIUS_MULT = 3.0
REACQUIRE_CHANGED_PIXELS = 1000     # you tune
REACQUIRE_DIFF_THRESHOLD = DIFF_THRESHOLD

EXIT_STABLE_FRAMES = 2             # <-- NEW: need 4 consecutive "exit-change" frames

# Exports
OUT_CSV  = "distance_trace.csv"
OUT_PLOT = "distance_plot.png"
OUT_TIMES = "inbox_times.csv"


# -----------------------------
# State
# -----------------------------
state = {
    "phase": "WAIT_SQUIRREL",
    "squirrel_center": None,
    "entrance_point": None,
    "prev_gray": None,
    "frame_idx": 0,

    "dist_rows": [],

    "in_box": False,
    "reacquire_mode": False,

    "in_counter": 0,

    # timer in VIDEO TIME
    "enter_time_s": None,
    "enter_frame": None,

    # exit confirmation in Mode 2
    "exit_counter": 0,
    "last_exit_center": None,

    # store final intervals
    "timer_rows": [],  # enter_frame, exit_frame, enter_time_s, exit_time_s, duration_s
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_circular_mask(h, w, center, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = center
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def compute_change_centroid(prev_gray, gray_now, mask, diff_threshold, min_changed_pixels):
    diff = cv2.absdiff(prev_gray, gray_now)
    diff_roi = cv2.bitwise_and(diff, diff, mask=mask)

    _, diff_bin = cv2.threshold(diff_roi, diff_threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    changed_count = int(cv2.countNonZero(diff_bin))
    if changed_count < min_changed_pixels:
        return changed_count, None, diff_bin

    M = cv2.moments(diff_bin, binaryImage=True)
    if M["m00"] == 0:
        return changed_count, None, diff_bin

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return changed_count, (cx, cy), diff_bin


def distance_px(p1, p2):
    if p1 is None or p2 is None:
        return None
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.sqrt(dx * dx + dy * dy))


def draw_overlay(frame, roi_center, roi_radius, entrance_pt, diff_bin, text_lines, extra_circles=None):
    overlay = frame.copy()

    if roi_center is not None:
        cv2.circle(overlay, roi_center, roi_radius, (0, 0, 255), 2)
        cv2.circle(overlay, roi_center, 4, (0, 0, 255), -1)

    if entrance_pt is not None:
        cv2.circle(overlay, entrance_pt, 6, (255, 255, 0), -1)
        cv2.putText(
            overlay, "entrance", (entrance_pt[0] + 8, entrance_pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

    if extra_circles:
        for (center, radius, color, thickness) in extra_circles:
            cv2.circle(overlay, center, radius, color, thickness)

    # Optional: visualize changed pixels
    # if diff_bin is not None:
    #     overlay[diff_bin > 0] = (0, 255, 0)

    y = 40
    for line in text_lines:
        cv2.putText(
            overlay, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
        y += 35

    return overlay


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state["phase"] == "WAIT_SQUIRREL":
        state["squirrel_center"] = (x, y)
        state["phase"] = "WAIT_ENTRANCE"
        state["prev_gray"] = None
        print(f"[Squirrel selected] {state['squirrel_center']}")

    elif state["phase"] == "WAIT_ENTRANCE":
        state["entrance_point"] = (x, y)
        print(f"[Entrance selected] {state['entrance_point']} (press Enter to confirm)")


def export_distance_plot(csv_path, plot_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[Plot] No data to plot.")
        return

    plt.figure()
    plt.plot(df["time_s"], df["dist_px"])
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (px)")
    plt.title("Squirrel-to-Entrance Distance (pixel)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


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
        time_s = state["frame_idx"] / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

        key = 0

        # -------------------------
        # Phase 1: select squirrel
        # -------------------------
        if state["phase"] == "WAIT_SQUIRREL":
            overlay = draw_overlay(frame, None, ROI_RADIUS, None, None, ["Click squirrel to start"])
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
                text_lines=["Select entrance (click) then press Enter to confirm"]
            )
            cv2.imshow(WINDOW, overlay)

            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):  # Enter
                if state["entrance_point"] is None:
                    print("[Warn] Please click entrance first.")
                else:
                    state["phase"] = "TRACKING"
                    state["prev_gray"] = gray.copy()
                    state["in_box"] = False
                    state["reacquire_mode"] = False
                    state["in_counter"] = 0
                    state["exit_counter"] = 0
                    state["enter_time_s"] = None
                    state["enter_frame"] = None
                    print("[Tracking started]")

        # -------------------------
        # Phase 3: tracking
        # -------------------------
        else:
            entrance_pt = state["entrance_point"]
            if state["prev_gray"] is None:
                state["prev_gray"] = gray.copy()

            # =========================================
            # MODE 2 (Reacquire / Exit detection)
            # =========================================
            if state["reacquire_mode"] and entrance_pt is not None:
                big_r = int(ROI_RADIUS * REACQUIRE_RADIUS_MULT)
                big_mask = make_circular_mask(h, w, entrance_pt, big_r)

                changed_count, new_center, diff_bin = compute_change_centroid(
                    state["prev_gray"], gray, big_mask,
                    diff_threshold=REACQUIRE_DIFF_THRESHOLD,
                    min_changed_pixels=REACQUIRE_CHANGED_PIXELS
                )

                state["prev_gray"] = gray.copy()

                # --- robust exit: require 4 consecutive frames with enough change
                if new_center is not None:
                    state["exit_counter"] += 1
                    state["last_exit_center"] = new_center
                else:
                    state["exit_counter"] = 0
                    state["last_exit_center"] = None

                lines = [
                    f"STATE: IN BOX | timer: {(time_s - state['enter_time_s']):.2f}s" if state["enter_time_s"] else "STATE: IN BOX",
                    f"MODE2 exit_check: {state['exit_counter']}/{EXIT_STABLE_FRAMES} | changed_px={changed_count}"
                ]

                extra = [(entrance_pt, big_r, (255, 0, 0), 2)]
                overlay = draw_overlay(frame, state["squirrel_center"], ROI_RADIUS, entrance_pt, diff_bin, lines, extra_circles=extra)

                # If enough consecutive exit frames -> exit confirmed
                if state["exit_counter"] >= EXIT_STABLE_FRAMES and state["last_exit_center"] is not None:
                    exit_center = state["last_exit_center"]

                    # finalize timing
                    if state["enter_time_s"] is not None and state["enter_frame"] is not None:
                        duration_s = time_s - state["enter_time_s"]
                        state["timer_rows"].append({
                            "enter_frame": state["enter_frame"],
                            "exit_frame": state["frame_idx"],
                            "enter_time_s": float(state["enter_time_s"]),
                            "exit_time_s": float(time_s),
                            "duration_s": float(duration_s),
                        })
                        print(f"[EXIT CONFIRMED] duration={duration_s:.2f}s")

                    # jump tracking back to new center
                    nx = clamp(exit_center[0], 0, w - 1)
                    ny = clamp(exit_center[1], 0, h - 1)
                    state["squirrel_center"] = (nx, ny)

                    # reset box mode
                    state["in_box"] = False
                    state["reacquire_mode"] = False
                    state["in_counter"] = 0
                    state["exit_counter"] = 0
                    state["enter_time_s"] = None
                    state["enter_frame"] = None
                    state["last_exit_center"] = None

                    print(f"[Back to tracking] new center={state['squirrel_center']}")

                cv2.imshow(WINDOW, overlay)
                key = cv2.waitKey(RUN_DELAY_MS) & 0xFF

            # =========================================
            # NORMAL TRACKING MODE
            # =========================================
            else:
                roi_center = state["squirrel_center"]
                mask = make_circular_mask(h, w, roi_center, ROI_RADIUS)

                changed_count, new_center, diff_bin = compute_change_centroid(
                    state["prev_gray"], gray, mask,
                    diff_threshold=DIFF_THRESHOLD,
                    min_changed_pixels=MIN_CHANGED_PIXELS
                )

                if new_center is not None:
                    nx = clamp(new_center[0], 0, w - 1)
                    ny = clamp(new_center[1], 0, h - 1)
                    state["squirrel_center"] = (nx, ny)

                state["prev_gray"] = gray.copy()

                dist = distance_px(state["squirrel_center"], entrance_pt)

                # If close enough => show/log as 0
                if dist is not None and dist <= IN_BOX_ZERO_DIST_PX:
                    dist_for_log = 0.0
                else:
                    dist_for_log = dist

                # log distance
                if dist_for_log is not None:
                    state["dist_rows"].append({
                        "frame": state["frame_idx"],
                        "time_s": time_s,
                        "dist_px": float(dist_for_log),
                        "changed_px": changed_count,
                        "cx": state["squirrel_center"][0],
                        "cy": state["squirrel_center"][1],
                        "ex": entrance_pt[0] if entrance_pt else None,
                        "ey": entrance_pt[1] if entrance_pt else None,
                        "in_box": int(state["in_box"]),
                        "reacquire": int(state["reacquire_mode"]),
                    })

                # --- enter box detection (EARLIER threshold) ---
                if not state["in_box"]:
                    if dist is not None and dist <= IN_BOX_ENTER_DIST_PX:
                        state["in_counter"] += 1
                    else:
                        state["in_counter"] = max(0, state["in_counter"] - 1)

                    if state["in_counter"] >= STABLE_FRAMES_IN:
                        state["in_box"] = True
                        state["reacquire_mode"] = True
                        state["enter_time_s"] = time_s
                        state["enter_frame"] = state["frame_idx"]
                        state["exit_counter"] = 0
                        state["last_exit_center"] = None
                        print(f"[IN BOX] frame={state['frame_idx']} dist={dist:.1f}px -> Mode2 ON")

                # text
                lines = []
                if dist_for_log is None:
                    lines.append("dist: n/a")
                else:
                    lines.append(f"dist_px: {dist_for_log:.1f} | changed_px: {changed_count}")

                if state["in_box"]:
                    t = 0.0 if state["enter_time_s"] is None else (time_s - state["enter_time_s"])
                    lines.append(f"STATE: IN BOX | timer: {t:.2f}s")
                else:
                    lines.append("STATE: OUTSIDE")

                overlay = draw_overlay(frame, state["squirrel_center"], ROI_RADIUS, entrance_pt, diff_bin, lines)
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
            state["reacquire_mode"] = False
            state["in_counter"] = 0
            state["exit_counter"] = 0
            state["enter_time_s"] = None
            state["enter_frame"] = None
            state["last_exit_center"] = None

    cap.release()
    cv2.destroyAllWindows()

    # -----------------------------
    # Exports
    # -----------------------------
    if state["dist_rows"]:
        df = pd.DataFrame(state["dist_rows"])
        df.to_csv(OUT_CSV, index=False)
        export_distance_plot(OUT_CSV, OUT_PLOT)
        print("Saved:", OUT_CSV)
        print("Saved:", OUT_PLOT)
    else:
        print("No distance data collected.")

    if state["timer_rows"]:
        pd.DataFrame(state["timer_rows"]).to_csv(OUT_TIMES, index=False)
        print("Saved:", OUT_TIMES)


if __name__ == "__main__":
    main()
