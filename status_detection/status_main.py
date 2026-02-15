import cv2
import numpy as np
from entry import detect_entry
from color_histo import draw_histogram_on_frame, squirrel_color_mask

# -----------------------------
# Setup & entrance detection
# -----------------------------
video = cv2.VideoCapture("../data/video03.2.mp4")
ret, frame = video.read()

if not ret:
    print("Could not read first frame")
    exit()

# Detect entrance + ring radii from first frame
circles = detect_entry(frame)
if circles is None:
    exit("No entrance detected.")

cx, cy = circles["center"]
r1 = circles["entrance"]   # inner
r2 = circles["half"]       # middle
r3 = circles["full"]       # outer

h, w = frame.shape[:2]

# -----------------------------
# Build masks for 3 regions
# -----------------------------
mask_inner  = np.zeros((h, w), dtype=np.uint8)
mask_middle = np.zeros((h, w), dtype=np.uint8)
mask_outer  = np.zeros((h, w), dtype=np.uint8)

# inner circle
cv2.circle(mask_inner, (cx, cy), r1, 255, -1)

# middle ring = r1..r2
tmp = np.zeros((h, w), dtype=np.uint8)
cv2.circle(tmp, (cx, cy), r2, 255, -1)
mask_middle = cv2.subtract(tmp, mask_inner)

# outer ring = r2..r3
tmp2 = np.zeros((h, w), dtype=np.uint8)
cv2.circle(tmp2, (cx, cy), r3, 255, -1)
mask_outer = cv2.subtract(tmp2, tmp)

# -----------------------------
# Thresholds
# -----------------------------
# motion thresholds
threshold_inner  = 5
threshold_middle = 4
threshold_outer  = 3

# “calm” thresholds to reset after full entry
reset_inner  = 2.0
reset_middle = 1.5
reset_outer  = 1.0
color_reset_limit = 2000    # max squirrel-color pixels allowed when “calm”

# color-based detection in inner region
inner_color_on_threshold = 5000  # squirrel-color pixels = squirrel near entrance

# state flags
full_mode = False        # latched "FULL ENTRY" state
had_full_entry = False   # tracks if we ever had a full entry before

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = video.read()
    if not ret:
        break

    # --- motion / diff ---
    gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_now = cv2.GaussianBlur(gray_now, (5, 5), 0)

    diff = cv2.absdiff(prev_gray, gray_now)
    prev_gray = gray_now

    diff_inner  = cv2.mean(diff, mask=mask_inner)[0]
    diff_middle = cv2.mean(diff, mask=mask_middle)[0]
    diff_outer  = cv2.mean(diff, mask=mask_outer)[0]

    # --- squirrel color in inner ring ---
    col_inner_mask  = squirrel_color_mask(frame, mask_inner)
    col_inner_count = int(col_inner_mask.sum() / 255)  # white pixels = fur-ish

    # convenience booleans
    inner_active  = (diff_inner  > threshold_inner)  or (col_inner_count > inner_color_on_threshold)
    middle_active = (diff_middle > threshold_middle)
    outer_active  = (diff_outer  > threshold_outer)

    # -----------------------------
    # State machine
    # -----------------------------
    state  = "No movement"
    colors = [(100,100,100), (100,100,100), (100,100,100)]  # default grey

    if full_mode:
        # We *stay* in FULL ENTRY until everything is calm again
        calm_motion = (
            diff_inner  < reset_inner  and
            diff_middle < reset_middle and
            diff_outer  < reset_outer
        )
        calm_color = (col_inner_count < color_reset_limit)

        if calm_motion and calm_color:
            # scene is calm again -> unlock full_mode
            full_mode = False
            state = "Reset – waiting for next squirrel"
        else:
            # still consider squirrel as “inside” (latched)
            state = "🔴 STILL FULL ENTRY"
            colors[2] = (0, 0, 255)

    else:
        # Normal detection mode

        # Full entry has highest priority
        if outer_active:
            state = "🔴 FULL ENTRY"
            colors[2] = (0, 0, 255)
            full_mode = True         # latch
            had_full_entry = True    # remember we had one
        # 2) Half entry in middle ring
        elif middle_active:
            state = "🟡 Half Entry"
            colors[1] = (0, 255, 255)
        # 3) Only inner region active
        elif inner_active:
            # If we never had full entry yet: it's probably just curiosity / head
            if not had_full_entry:
                state = "🟢 Curious (Head at entrance)"
            else:
                # We *did* have a previous full entry and now only inner is active:
                # -> squirrel has left box but is still sitting in front of entrance
                state = "🟢 Squirrel in front of box"
            colors[0] = (0, 255, 0)
        else:
            state = "No movement"

    print(
        f"Inner:{diff_inner:.1f}  "
        f"Middle:{diff_middle:.1f}  "
        f"Outer:{diff_outer:.1f}  "
        f"ColorInner:{col_inner_count}  ->  {state}"
    )

    # -----------------------------
    # Drawing + histogram overlay
    # -----------------------------
    overlay = frame.copy()

    cv2.circle(overlay, (cx, cy), r1, colors[0], 3)
    cv2.circle(overlay, (cx, cy), r2, colors[1], 3)
    cv2.circle(overlay, (cx, cy), r3, colors[2], 3)

    cv2.putText(
        overlay, state, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    # histogram of colors in inner region at bottom of frame
    overlay_with_hist = draw_histogram_on_frame(overlay, mask_inner)
    cv2.imshow("Squirrel Tracker", overlay_with_hist)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()