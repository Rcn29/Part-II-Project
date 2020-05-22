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

camera_width=1024
camera_height=768
folderName='MorningBigHandTest/'
sampleSaved=False

def SaveImg(filename, imageArray):
    imageArray = cv2.flip(imageArray, 0);
    cv2.imwrite(filename, imageArray)

camera = PiCamera()
camera.resolution = (camera_width, camera_height)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(camera_width, camera_height))
background = PiRGBArray(camera, size=(camera_width, camera_height))
# allow the camera to warmup
time.sleep(0.5)
# Now fix the values
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g
camera.capture(background,format='bgr', use_video_port=True)
print("Bg Done")
SaveImg('Background.jpg',background.array)
#cv2.imshow("Background",grayground)
time.sleep(4)
cv2.destroyAllWindows()
index=0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
#while(True):
#    camera.capture(rawCapture,format='rgb')
#    image = rawCapture.array
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    index+=1
    print("Captured " + str(index))
    # show the frame
    #cv2.imshow("Frame", image)
    #SaveImg(folderName+'Frame'+ str(index) +'.jpg',image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if index==1:
        break