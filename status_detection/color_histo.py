import cv2
import numpy as np

def draw_histogram_on_frame(frame, mask, bins=180):
    """
    Compute a hue histogram inside `mask` and draw it as a bar chart
    below the frame. Returns the combined image.
    """

    # --- 1) Convert to HSV and compute histogram on Hue channel ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]

    hist = cv2.calcHist(
        [h_channel],   # images
        [0],           # channels (Hue)
        mask,          # mask
        [bins],        # histogram size
        [0, 180]       # Hue range
    )

    # Normalize to [0, 1] for drawing
    hist = cv2.normalize(hist, None, alpha=0, beta=1,
                         norm_type=cv2.NORM_MINMAX).flatten()

    # --- 2) Create an empty image for the histogram ---
    hist_h = 100                         # height of histogram image
    hist_w = frame.shape[1]              # make same width as frame
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    bin_w = max(1, hist_w // bins)       # pixel width of each bin

    # --- 3) Draw bars ---
    for i, v in enumerate(hist):
        x1 = int(i * bin_w)
        x2 = int((i + 1) * bin_w)
        # height of the bar
        bar_h = int(v * (hist_h - 1))
        y1 = hist_h - 1
        y2 = hist_h - 1 - bar_h

        # Make sure coordinates are ints and in range
        x1 = max(0, min(hist_w - 1, x1))
        x2 = max(0, min(hist_w - 1, x2))
        y1 = max(0, min(hist_h - 1, y1))
        y2 = max(0, min(hist_h - 1, y2))

        cv2.rectangle(
            hist_img,
            (x1, y2),
            (x2, y1),
            (0, 255, 0),
            thickness=-1
        )

    # --- 4) Stack frame + histogram vertically ---
    combined = np.vstack((frame, hist_img))
    return combined


def squirrel_color_mask(frame, region_mask=None):
    """
    Returns a binary mask of pixels whose color looks like squirrel fur.
    Optionally restricts it to a region_mask (inner / middle / outer ring).
    """

    # BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- rough reddish/brown range in HSV ---
    # tune these numbers!
    # H: 5..25   (orange/brown)
    # S: 40..255 (avoid gray/white wall)
    # V: 40..255 (avoid very dark noise)
    lower = np.array([5, 40, 40], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    color_mask = cv2.inRange(hsv, lower, upper)

    # optionally restrict to circular region (inner / middle / outer)
    if region_mask is not None:
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=region_mask)

    return color_mask