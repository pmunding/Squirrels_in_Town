# in this approach we use opencv's built in background substraction method MOG2 to substract the moving objects from the background 
# Source - https://stackoverflow.com/a

import cv2
import cv2 as cv
import numpy as np

# Load video
capture = cv2.VideoCapture('../data/video02.mp4')

# Define video writer
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*"DIB ")
video = cv2.VideoWriter('export/output.avi', fourcc, 30,size)
fgbg= cv2.createBackgroundSubtractorMOG2()


while True:
    ret, img = capture.read()
    if ret==True:
        # Apply background subtractor to get the foreground mask
        fgmask = fgbg.apply(img)
        video.write(fgmask)
        cv2.imshow('forehead',fgmask)

    if(cv2.waitKey(27)!=-1):
        break

capture.release()
video.release()
cv2.destroyAllWindows()

