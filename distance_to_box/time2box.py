# This module implements a simple ROI tracker based on pixel change detection.
import cv2
import numpy as np

# --- Video and window settings ---
VIDEO_PATH = "trimshort.mp4"  # <-- anpassen
WINDOW = "Squirrel ROI Tracker | click=init | q=quit | r=reset"

#--- Parameters ---
SLOW_DELAY_MS = 250  # vor dem Klick sehr langsam
RUN_DELAY_MS  = 100   # nach dem Klick normaler/halbwegs flüssig

# --- ROI / Maske ---
ROI_RADIUS = 80          # Circular ROI um den Klickpunkt (Pixel)
MIN_CHANGED_PIXELS = 200 # threshold: wie viele Pixel müssen sich ändern, damit wir es als Bewegung werten? (höher = weniger sensitiv)

# --- Change Detection ---
DIFF_THRESHOLD = 18      # Pixel-value-Difference, to count as "changed" 
BLUR_K = 5               


state = {
    "initialized": False,
    "center": None,        
    "prev_gray": None,     
}

# helper functions
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# helper function to create a filled mask from a contour (optional, not used in main code)
def make_circular_mask(h, w, center, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = center
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask

# helper function to extract a sub-image (ROI) around a center point, with boundary checks
def compute_change_centroid(prev_gray, gray_now, mask):
    """
    Gibt zurück:
      changed_count, centroid (x,y) oder None, diff_mask (binär)
    """
    diff = cv2.absdiff(prev_gray, gray_now)
    # nur in ROI arbeiten
    diff_roi = cv2.bitwise_and(diff, diff, mask=mask)

    # binarisieren
    _, diff_bin = cv2.threshold(diff_roi, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # kleines Cleanup (optional, hilft gegen Pixelrauschen)
    kernel = np.ones((3, 3), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    changed_count = int(cv2.countNonZero(diff_bin))
    if changed_count < MIN_CHANGED_PIXELS:
        return changed_count, None, diff_bin

    # Schwerpunkt (Centroid) der geänderten Pixel
    M = cv2.moments(diff_bin, binaryImage=True)
    if M["m00"] == 0:
        return changed_count, None, diff_bin

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return changed_count, (cx, cy), diff_bin


# --- Main Code ---
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and not state["initialized"]:
        state["initialized"] = True
        state["center"] = (x, y)
        print(f"[Init] center set to: {state['center']}")


def draw_overlay(frame, center, radius, diff_bin=None, info_text=""):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    if center is not None:
        cv2.circle(overlay, center, radius, (0, 0, 255), 2)
        cv2.circle(overlay, center, 4, (0, 0, 255), -1)

    if diff_bin is not None:
        # Change-Pixel as green highlighten
        overlay[diff_bin > 0] = (0, 255, 0)

    if info_text:
        cv2.putText(overlay, info_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return overlay

# Compute minimum distance from squirrel pixels to entrance pixels using distance transform.
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    frozen_init_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]

        # preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

        key = 0

        # -------------------------
        # Phase 1: waiting for click to initialize ROI
        # -------------------------
        if not state["initialized"]:
            # sehr langsames Abspielen
            overlay = draw_overlay(frame, None, ROI_RADIUS, None, "Click squirrel to start")
            cv2.imshow(WINDOW, overlay)
            key = cv2.waitKey(SLOW_DELAY_MS) & 0xFF

        # -------------------------
        # Phase 2: Tracking
        # -------------------------
        else:
            # Click on the squirrel, now we track the ROI based on pixel changes
            if state["prev_gray"] is None:
                state["prev_gray"] = gray.copy()
                frozen_init_frame = frame.copy()

                # stop and show initial ROI until next click
                mask = make_circular_mask(h, w, state["center"], ROI_RADIUS)
                overlay = draw_overlay(frozen_init_frame, state["center"], ROI_RADIUS, None, "Initialized ROI")
                cv2.imshow(WINDOW, overlay)
                key = cv2.waitKey(0) & 0xFF  # warten bis Taste gedrückt (optional)
            else:
                # update ROI center based on pixel changes
                mask = make_circular_mask(h, w, state["center"], ROI_RADIUS)

                changed_count, new_center, diff_bin = compute_change_centroid(
                    state["prev_gray"], gray, mask
                )

                # If changed_count is above threshold, update center to new_center
                if new_center is not None:
                    # clamp in Bildgrenzen (normalerweise schon ok)
                    nx = clamp(new_center[0], 0, w - 1)
                    ny = clamp(new_center[1], 0, h - 1)
                    state["center"] = (nx, ny)

                # update previous frame for next iteration
                state["prev_gray"] = gray.copy()

                info = f"changed_px={changed_count} | center={state['center']}"
                overlay = draw_overlay(frame, state["center"], ROI_RADIUS, diff_bin, info)
                cv2.imshow(WINDOW, overlay)
                key = cv2.waitKey(RUN_DELAY_MS) & 0xFF

        # --- Controls ---
        if key == ord("q") or key == 27:
            break
        if key == ord("r"):
            state["initialized"] = False
            state["center"] = None
            state["prev_gray"] = None
            frozen_init_frame = None
            print("[Reset]")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
