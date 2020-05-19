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

def stopCameraAutos(camera):
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g

inputW = 640
inputH = 368

camera = PiCamera()
camera.resolution = (inputW, inputH)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(inputW,inputH))
# allow the camera to warmup
#camera.start_preview() #DO NOT DO THIS
camera.capture(rawCapture, format="bgr", use_video_port=True)
rawCapture.truncate(0)
time.sleep(5.0)

cv2.namedWindow("Frame")
cannyMin = 80
cannyMax = 140
def trackMin(v) :
    global cannyMin
    cannyMin = v
def trackMax(v) :
    global cannyMax
    cannyMax = v
#cv2.createTrackbar("Canny Min", "Frame" , cannyMin, 500, trackMin)
#cv2.createTrackbar("Canny Max", "Frame" , cannyMax, 500, trackMax)

stopCameraAutos(camera)

minT = [1,2,3]
maxT = [1,2,3]
flexMin = 0.9
flexMax = 1.1
adjusting = True

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = np.array(frame.array)
    image =cv2.flip(image,-1)
    #image =cv2.GaussianBlur(image, (7,7), 0)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    t0=time.time()
    
    cx = int(inputW * 0.5)
    cy = int(inputH * 0.5)
    
    x1 = cx-15
    x2 = cx+15
    y1 = cy-15
    y2 = cy+15
    if adjusting:
        for i in range(3):
            imageH = imageHSV[y1:y1+30,x1:x1+30, i]
            minV,maxV,_,_ = cv2.minMaxLoc( imageH)
            minT[i] = minV * flexMin
            maxT[i] = maxV * flexMax
    
    imageThresh = cv2.inRange(imageHSV, tuple(minT), tuple(maxT))
    #imageThresh = cv2.GaussianBlur(imageThresh, (5,5), 0)
    imageThresh = cv2.cvtColor(imageThresh, cv2.COLOR_GRAY2BGR)
    
    image = cv2.addWeighted( image, 0.5, imageThresh, 0.5, 0.0)
    
    if adjusting:
        cv2.rectangle(image, (x1,y1),(x2,y2), (0,200,0), 2)
    cv2.putText( image,
     "Hue " + str(minT) + " " + str(maxT), (40, 40),
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     1, #font size
     (209, 80, 0), #font color
     2)
    
    t1=time.time()
    print(str(t1-t0))
    #image=cv2.Canny(image,50,120)
    # show the frame
    cv2.imshow("Frame", image)
    
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    if key == ord("s"):
        adjusting = not adjusting
    if key == ord("q"):
        break
cv2.destroyAllWindows()

