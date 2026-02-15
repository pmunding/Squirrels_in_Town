# installing cvzone library for background removal
# pip install cvzone
# second installing the mediapipe library
import cv2
import cvzone
# importing SelfiSegmentation module from cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

source = cv2.VideoCapture('../data/video_with_change.mp4')

# creating object for SelfiSegmentation
segmentor = SelfiSegmentation() 

while True:
    
    ret, img = source.read()
    if ret==True:
        # using removeBG function to remove background from image
        # and replace it with a solid color (here white)
        imgOut = segmentor.removeBG(img, (255, 255, 255), threshold=0.8)
        
        cv2.imshow('Image', imgOut)

    if(cv2.waitKey(27)!=-1):
        break

