import cv2
import numpy as np

# Load Video
cap = cv2.VideoCapture("../data/video_with_change.mp4")
first = cv2.VideoCapture("../data/video_bg.mp4")

# Read first frame → Background Reference
ret, bg = first.read()
if not ret:
    print("Could not read video")
    exit()

bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# Optional: reduce noise in reference
bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Difference to background
    diff = cv2.absdiff(gray, bg_gray)

    # Threshold → what is foreground?
    _, fgMask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Clean mask (remove noise)
    kernel = np.ones((5,5), np.uint8)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Extract only the moving foreground
    fg = cv2.bitwise_and(frame, frame, mask=fgMask)

    # Show result
    cv2.imshow("Original", frame)
    cv2.imshow("Foreground Mask", fgMask)
    cv2.imshow("Foreground Only", fg)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()