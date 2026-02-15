# in this example of bachground estimation we use OpenCV to read a video file
# than we select random frames from the video from which we will extract the background
# and compute the median frame to estimate the background.
# Source: https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/

import numpy as np
import cv2
from skimage import data, filters
 
# Open Video
cap = cv2.VideoCapture('../exports/demo.mp4')
 
# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25) # the quality increseses with the number of frames
 
# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
 
# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
 
# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)