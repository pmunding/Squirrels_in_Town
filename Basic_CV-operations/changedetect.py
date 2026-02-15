# turning into gray-scale video: https://www.geeksforgeeks.org/machine-learning/converting-color-video-to-grayscale-using-opencv-in-python/

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# loading the video
source = cv2.VideoCapture('data/squirrel_videos/trim04.mp4')

# current frame and previous frame
ret, prev_frame = source.read()
ret2, now_frame = source.read()

# storing video properties
frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(source.get(cv2.CAP_PROP_FPS))

# safe output video setup
out = cv2.VideoWriter(
    'data/change_folder/change1010_TrepS_02_in (5.2).mp4',       # output file name        
    cv2.VideoWriter_fourcc(*'mp4v'),         
    fps,                                     
    (frame_width, frame_height)              
)


# Thrashold for frame difference
CHANGE_THRESHOLD = 1.45

# create a list to store the change for each frame
change = []

# counter saved frames
frame_index = 0

# counter for saved pixels for each frame

# while ret2 is True (there is a next frame)
while ret2:

    # convert into gray-scale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    now_gray = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)

    # reduce noise with Gaussian blur
    prev_gray = cv2.GaussianBlur(prev_gray, (5,5), 0)
    now_gray = cv2.GaussianBlur(now_gray, (5,5), 0)

    # substract the two frames
    diff = cv2.absdiff(prev_gray, now_gray)
    mean_diff = np.mean(diff)

    # if the mean difference is above the threshold, save the frame
    if mean_diff > CHANGE_THRESHOLD:
        out.write(now_frame)
        
        print(f"Frame {frame_index}: saved (mean_diff={mean_diff:.2f})")
    else:
        print(f"Frame {frame_index}: skipped (mean_diff={mean_diff:.2f})")

    # display
    cv2.imshow("Difference", diff)

    # swapping previous and now frame
    prev_frame = now_frame.copy()

    # reading the next frame
    ret2, now_frame = source.read()
    frame_index += 1

    # show difference
    cv2.imshow("Difference", diff)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# terminating the video
source.release()
out.release()
cv2.destroyAllWindows()


