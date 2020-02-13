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

camera_width=640
camera_height=480

camera=PiCamera()
camera.resolution=(camera_width,camera_height)
camera.framerate=15
rawCapture=PiRGBArray(camera)
time.sleep(0.1)
camera.capture(rawCapture,format="bgr")
image=rawCapture.array
cv2.imshow("Image",image)
print(image[23][24])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(gray[23][24])