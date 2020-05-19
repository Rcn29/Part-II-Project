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

def SaveImg(filename, imageArray):
    imageArray = cv2.flip(imageArray, 0);
    cv2.imwrite(filename, imageArray)

camera_width=1920
camera_height=1080

camera = PiCamera()
camera.resolution = (camera_width, camera_height)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(camera_width, camera_height))
time.sleep(0.5)

index=0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    index+=1
    print("Captured "+str(index))
    
    cv2.imwrite("CalibrationSerban/Frame"+str(index)+".jpg",image)
    cv2.imshow("Image",image)
    cv2.waitKey(500)
    rawCapture.truncate(0)
    if index==50:
        break
    