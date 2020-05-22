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
import MotionMask
from pathlib import Path
from math import sqrt

camera = PiCamera()
camera.resolution= (1280, 768)
camera.framerate = 16
rawCapture=PiRGBArray(camera,size=(1280,768))
camera.capture(rawCapture, format="bgr", use_video_port=True)
rawCapture.truncate(0)
time.sleep(2)
ok=False

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image=np.array(frame.array)
    image=cv2.flip(image,-1)
    imageHSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    if ok:
        prevFrame=cv2.resize(MotionMask.prevImage,(1280,768))
    MotionMask.updateMotion(image)
    ok=True
    img=image.copy()
    img=MotionMask.apply(img)
    cv2.imshow("MotionMasked",img)
    key=cv2.waitKey(1)&0xFF
    if key == ord("s"):
       cv2.imwrite("MotionMaskInitialFrame.jpg",prevFrame)
       cv2.imwrite("MotionMaskSecondFrame.jpg",image)
       cv2.imwrite("MotionMaskResult.jpg",img)
    if key == ord("q"):
        break
    rawCapture.truncate(0)