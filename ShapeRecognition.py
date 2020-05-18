import sys
#print(sys.path)
import os
#print(os.getcwd())
os.chdir('/home/pi')
#print(os.getcwd())
sys.path.append('')
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import glob
from matplotlib import pyplot as plt

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 1
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    image = frame.array
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #gray=cv2.Canny(gray,30,90)
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,100)
    output=gray.copy()
    
    if circles is not None:
        circles=np.round(circles[0,:]).astype("int")
        print("Found circles")
        for (x, y, r) in circles:
            cv2.circle(output,(x,y),r,(0,255,0),4)
            cv2.rectangle(output,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
    
    cv2.imshow("output",output)
    cv2.waitKey(0)
    rawCapture.truncate(0)
    #if key==ord("q"):
    #    break
    