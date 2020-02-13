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

camera_width=1280
camera_height=1024

camera = PiCamera()
camera.resolution = (camera_width, camera_height)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(camera_width, camera_height))
# allow the camera to warmup
time.sleep(0.1)
background=PiRGBArray(camera, size=(camera_width, camera_height))
camera.capture(background,format="bgr")
bgr=background.array
grayground=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
cv2.imwrite("Background.jpg",grayground)
cv2.imshow("Background",grayground)
time.sleep(5)
cv2.destroyAllWindows()
index=0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    index+=1
    image = frame.array
    # show the frame
    cv2.imshow("Frame", image)
    cv2.imwrite("FrameNumber"+str(index)+".jpg",image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if index==31:
        break