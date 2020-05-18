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

def CannifyImage(image,show):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,30,90)
    if show:
        plt.subplot(121),plt.imshow(gray,cmap='gray')
        plt.title('Original grayscale image'),plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap='gray')
        plt.title('Edge image'),plt.xticks([]),plt.yticks([])
        plt.show()

#todo: keep motion histogram in half/quarter resolution so it's faster

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)

backSub=cv2.createBackgroundSubtractorMOG2()
background = None
prevImage = None
motionHist = None
dilateKernel = np.ones((10,10), np.uint8)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    image = cv2.flip(image, 0)
    if motionHist is None:
        motionHist = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)

    t0=time.time()
    if True:
        #image = cv2.absdiff( image, background)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, motionMask = cv2.threshold(motionHist, 15, 255, cv2.THRESH_BINARY)
        image = motionMask & image 
        
        #image = cv2.GaussianBlur(image,(3,3),0)
        #ret, image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    else :
        image = backSub.apply(image)
        image = cv2.GaussianBlur(image,(3,3),0)            
        #ret, image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        
    cont, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, cont, -1, (127), 3)
    
    if prevImage is not None:
        motionImg = cv2.absdiff( frame.array, prevImage)
        motionImg = cv2.cvtColor(motionImg, cv2.COLOR_BGR2GRAY)
        motionImg = cv2.flip(motionImg, 0)
        t = 0.3
        motionHist = cv2.addWeighted( motionHist, 1.0 - t, motionImg, t, 0.0)
        motionHist = cv2.dilate(motionHist, dilateKernel)
    prevImage = frame.array.copy()
        
    t1=time.time()
    print(str(t1-t0))
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # keep bg image
        background = image.copy()
        # lock camera settings
        if True:
            camera.shutter_speed = camera.exposure_speed
            camera.exposure_mode = 'off'
            g = camera.awb_gains
            camera.awb_mode = 'off'
            camera.awb_gains = g
        print("Bg captured")
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    if key == ord("q"):
        break
