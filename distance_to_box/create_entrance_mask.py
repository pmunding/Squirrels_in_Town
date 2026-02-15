import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "sam_distance01.mp4"
OUT_VIDEO  = "video_with_distance.mp4"
OUT_CSV    = "distances.csv"

# Overlay-Farben (BGR!) – anpassen falls nötig
PINK_BGR = np.array([255, 0, 255], dtype=np.uint8)   # magenta/pink = squirrel
CYAN_BGR = np.array([255, 255, 0], dtype=np.uint8)   # cyan = entrance

TOL_PINK = 70
TOL_CYAN = 70

MIN_SQ_PIXELS = 500   # noise filter
K = 9                 # kernel size for morphology


def mask_from_overlay_color(frame_bgr, target_bgr, tol):
    """Binary mask where pixels are close to target BGR (uint8)."""
    frame_bgr = frame_bgr.astype(np.uint8)
    target_bgr = np.array(target_bgr, dtype=np.uint8)
    target_img = np.full_like(frame_bgr, target_bgr)

    diff = cv2.absdiff(frame_bgr, target_img)

    # sum channel diffs
    dist = (diff[:, :, 0].astype(np.int16) +
            diff[:, :, 1].astype(np.int16) +
            diff[:, :, 2].astype(np.int16))

    mask = (dist < (3 * tol)).astype(np.uint8) * 255
    return mask


def fill_mask(mask):
    """Fill blobs by contour fill."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 255, thickness=-1)
    return filled


def build_entrance_mask(frame):
    """Build entrance mask ONCE (from first frame)."""
    ent_raw = mask_from_overlay_color(frame, CYAN_BGR, TOL_CYAN)

    kernel = np.ones((K, K), np.uint8)

    # entrance is often only outline -> thicken then fill
    ent = cv2.dilate(ent_raw, kernel, iterations=2)
    ent = cv2.morphologyEx(ent, cv2.MORPH_CLOSE, kernel)
    ent = fill_mask(ent)

    return ent


def build_squirrel_mask(frame):
    """Build squirrel mask PER FRAME."""
    sq_raw = mask_from_overlay_color(frame, PINK_BGR, TOL_PINK)

    kernel = np.ones((K, K), np.uint8)

    sq = cv2.morphologyEx(sq_raw, cv2.MORPH_CLOSE, kernel)
    sq = fill_mask(sq)

    return sq


def min_distance_px(entrance_mask, squirrel_mask):
    """Return (dmin, squirrel_point, entrance_point) in pixels."""
    sq_count = int(np.sum(squirrel_mask > 0))
    if sq_count < MIN_SQ_PIXELS:
        return None, None, None, sq_count

    # distance transform: need zeros where entrance is, ones elsewhere
    # we want distance to entrance -> compute distances to nearest entrance pixel
    inv = cv2.bitwise_not(entrance_mask)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    ys, xs = np.where(squirrel_mask > 0)
    if xs.size == 0:
        return None, None, None, sq_count

    dvals = dist[ys, xs]
    idx = int(np.argmin(dvals))
    dmin = float(dvals[idx])
    sq_pt = (int(xs[idx]), int(ys[idx]))

    # find nearest entrance pixel (local search)
    r = max(5, int(np.ceil(dmin)) + 10)
    x0, y0 = sq_pt
    x1, x2 = max(0, x0 - r), min(entrance_mask.shape[1], x0 + r + 1)
    y1, y2 = max(0, y0 - r), min(entrance_mask.shape[0], y0 + r + 1)

    patch = entrance_mask[y1:y2, x1:x2]
    by, bx = np.where(patch > 0)
    if by.size == 0:
        box_pt = None
    else:
        bx_abs = bx + x1
        by_abs = by + y1
        dx = bx_abs - x0
        dy = by_abs - y0
        j = int(np.argmin(dx*dx + dy*dy))
        box_pt = (int(bx_abs[j]), int(by_abs[j]))

    return dmin, sq_pt, box_pt, sq_count


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --- read first frame and build entrance mask ONCE ---
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    entrance_mask = build_entrance_mask(first)
    entrance_count = int(np.sum(entrance_mask > 0))
    if entrance_count < 50:
        print("[WARN] Entrance mask looks empty. Check CYAN_BGR / TOL_CYAN.")

    rows = []
    frame_idx = 0

    # process first frame too
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        squirrel_mask = build_squirrel_mask(frame)
        dmin, sq_pt, box_pt, sq_count = min_distance_px(entrance_mask, squirrel_mask)

        overlay = frame.copy()

        # Draw entrance outline for orientation (optional)
        overlay[entrance_mask > 0] = (255, 255, 0)   # cyan-ish highlight
        overlay[squirrel_mask > 0] = (255, 0, 255)   # pink highlight

        if dmin is not None:
            cv2.putText(overlay, f"min_dist_px: {dmin:.1f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.circle(overlay, sq_pt, 6, (255, 0, 255), -1)
            if box_pt is not None:
                cv2.circle(overlay, box_pt, 6, (255, 255, 0), -1)
                cv2.line(overlay, sq_pt, box_pt, (0, 255, 255), 2)
        else:
            cv2.putText(overlay, "min_dist_px: n/a", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # debug counts so you see if mask is empty
        cv2.putText(overlay, f"sq_px: {sq_count}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        writer.write(overlay)

        rows.append({"frame": frame_idx, "min_dist_px": dmin, "sq_pixels": sq_count})
        frame_idx += 1

        cv2.imshow("Overlay", overlay)
        cv2.imshow("Entrance mask (fixed)", entrance_mask)
        cv2.imshow("Squirrel mask (per frame)", squirrel_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_VIDEO)
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
