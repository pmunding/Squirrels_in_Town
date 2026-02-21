# This module computes pixel distances from squirrel to entrance using OpenCV distance transform.
# It assumes you have binary masks for entrance and squirrel, which can be created using create_
import cv2
import numpy as np

# Parameters for color-based mask creation (tune if needed)
def preprocess_mask(mask):
    """Ensure mask is binary uint8 {0,255} and denoised a bit."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return mask

# Compute distance map from entrance mask, then compute stats for squirrel pixels.
def entrance_distance_map(entrance_mask):
    """
    Returns float32 dist map in pixels where each pixel = distance to entrance.
    OpenCV distanceTransform computes distance to nearest zero pixel,
    so we invert: entrance pixels -> 0, background -> 255.
    """
    entrance_mask = preprocess_mask(entrance_mask)
    inv = cv2.bitwise_not(entrance_mask)           # entrance becomes 0
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)  # float32
    return dist

# Compute distance from squirrel pixels to entrance using distance map.
def squirrel_to_entrance_distance(dist_map, squirrel_mask, stat="min"):
    squirrel_mask = preprocess_mask(squirrel_mask)
    ys, xs = np.where(squirrel_mask > 0)
    if len(xs) == 0:
        return None
    dvals = dist_map[ys, xs]
    if stat == "min":
        return float(np.min(dvals))
    if stat == "median":
        return float(np.median(dvals))
    if stat == "mean":
        return float(np.mean(dvals))
    raise ValueError("stat must be min/median/mean")

# Sometimes the entrance is just an outline, so we want to fill it to get a solid mask.
def closest_points(dist_map, squirrel_mask):
    """Return (x,y) on squirrel that is closest to entrance, and distance."""
    squirrel_mask = preprocess_mask(squirrel_mask)
    ys, xs = np.where(squirrel_mask > 0)
    if len(xs) == 0:
        return None, None, None
    dvals = dist_map[ys, xs]
    k = int(np.argmin(dvals))
    x, y = int(xs[k]), int(ys[k])
    return x, y, float(dvals[k])
