#In this approach, we try to separate the moving object from the background.
# First, we load a clean background frame from a second video. Then, 
# for each frame of the main video, we calculate the difference between 
# the current frame and that background. Pixels that change significantly are treated 
# as foreground — in this case, the squirrel.
# From this difference, we create a binary mask that highlights only the moving regions.
# We then clean this mask with morphological operations to remove noise.
# Using this mask, we extract the foreground from the original frame 
# and place it onto a completely white background. 
# Finally, the program displays the mask and the resulting image in real time.


import cv2 as cv
import numpy as np


cap = cv.VideoCapture('../data/video_with_change.mp4')

backg = cv.VideoCapture("../data/video_bg.mp4")


# Read first frame as background reference
ret, bgReference = backg.read()
if not ret:
    print("Could not read video")
    cap.release()
    exit(0)

bgReference = bgReference.copy()

while True:
    ret, img = cap.read()
    if not ret:
        break  # video finished

    # difference between current frame and background reference
    diff = cv.absdiff(img, bgReference)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    # Threshold to get moving/changed regions (foreground)
    _, fgMask = cv.threshold(gray, 15, 255, cv.THRESH_BINARY)

    # Optional: clean up mask
    kernel = np.ones((3, 3), np.uint8)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel, iterations=2)

    # --- Build white background + foreground composite ---

    # Foreground (where mask == 255)
    fg = cv.bitwise_and(img, img, mask=fgMask)

    # Inverted mask for background (255 where background, 0 where foreground)
    fgMask_inv = cv.bitwise_not(fgMask)

    # Pure white background image
    white_bg = np.full_like(img, 255)  # 255,255,255 everywhere

    # Apply background mask to white image
    bg = cv.bitwise_and(white_bg, white_bg, mask=fgMask_inv)

    # Combine white background + original foreground
    result = cv.add(bg, fg)

    # Show everything
    cv.imshow('Mask', fgMask)
    cv.imshow('Result (white background)', result)

    key = cv.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()