import sys
#print(sys.path)
import os
#print(os.getcwd())
os.chdir('/home/pi')
#print(os.getcwd())
sys.path.append('')
import cv2
import numpy as np
import time
import glob
from picamera.array import PiRGBArray
from picamera import PiCamera
"""
with PiCamera() as camera:
    camera.resolution=(1024,720)
    camera.framerate = 1
    time.sleep(2)
    output=np.empty((720,1024,3),dtype=np.uint8)
    camera.capture(output,'rgb')
"""
fname='/home/pi/Chessboard1.jpg'
img=cv2.imread(fname)
cv2.imshow('img',img)
cv2.waitKey(2000)