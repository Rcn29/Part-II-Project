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

def updateHistory(img, historyImg, t):
    return cv2.addWeighted( historyImg, 1.0 - t, img, t, 0.0)

#todo: keep motion histogram in half/quarter resolution so it's faster

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)

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

backSub=cv2.createBackgroundSubtractorMOG2()
prevImage = None
motionHist = None
prevImage16 = None
bgHist = None
useMotionMask=True
dilateKernel = np.ones((10,10), np.uint8)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    t0=time.time()
    image = cv2.flip(image, -1)
    imageH = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageH = imageH[:,:,2]
    image = imageH
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if motionHist is None:
        motionHist = image.copy()
    
    if prevImage is not None:
        motionImg = cv2.absdiff( image, prevImage)
        motionImg = cv2.GaussianBlur(motionImg,(5,5),0)
        motionHist = updateHistory(motionImg, motionHist, 0.2)
        """
        motionImg = cv2.absdiff( frame.array, prevImage)
        motionImg = cv2.cvtColor(motionImg, cv2.COLOR_BGR2GRAY)
        motionImg = cv2.flip(motionImg, 0)
        t = 0.3
        motionHist = cv2.addWeighted( motionHist, 1.0 - t, motionImg, t, 0.0)
        """
        motionHist = cv2.dilate(motionHist, dilateKernel)
        
    prevImage = image.copy()
    if useMotionMask:
        #image = cv2.absdiff( image, background)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #motion = cv2.absdiff(image, ( bgHist / 255).astype('uint8') )
        #ret, motionMask = cv2.threshold(motion, 50, 255, cv2.THRESH_BINARY)
        ret, motionMask = cv2.threshold(motionHist, 15, 255, cv2.THRESH_BINARY)
        #image = cv2.absdiff(image, bgHist)
        image = motionMask & image 

        #image = cv2.GaussianBlur(image,(3,3),0) #important. ajuta ca elimina din 
        
    else :
        image = cv2.GaussianBlur(image,(3,3),0)   
        image = backSub.apply(image)         
        #ret, image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        
    #todo: crop image to select only moving parts
    #image = cv2.Canny(image,cannyMin,cannyMax)
    #ret, image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    
    cont, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    fingers = []
    maxPts = 0
    maxIdx = -1
    for idx,points in enumerate(cont):
        if len(cont) > 50: break
        if maxPts < len(points):
            maxPts = len(points)
            maxIdx = idx
        """
        if len(points) < 2: continue
        if len(points) > 100: break
        isFinger = True
        length = 0.0
        for i,pt in enumerate(points):
            if i < 2:
                continue
            a = (points[i - 1] - points[i-2])[0]
            alen = np.linalg.norm(a)
            a = a / alen
            b = (points[i - 1] - points[i])[0]
            blen = np.linalg.norm(b)
            b = b / blen
            if abs(np.dot(a,b)) > 0.9:
                length += alen + blen
                isFinger = True
                break
        if length > 2 and length < 8:                
            #find finger-like point sequences
            #print(points)
            fingers.append(points)
        """
    print( len(cont), maxPts)
    #for idx,fgr in enumerate(cont):
    #    image = cv2.drawContours(image, cont, idx, (64 + 16 * idx), 2)
    #image = cv2.drawContours(image, cont, -1, (64), 1)
    #if maxIdx != -1:
    #    image = cv2.drawContours(image, cont, maxIdx, (128), 2)
    #lines = cv2.HoughLinesP(image, 4, np.pi / 180, 15, None, 2, 2)
    if False and lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image, (x1,y1), (x2,y2), (64), 2)
    
    #circles=cv2.HoughCircles(image,cv2.HOUGH_GRADIENT, 1.6, 20,param1=cannyMin*2,param2=35,maxRadius=20)
    #if circles is not None:
    #    circles=np.round(circles[0,:]).astype("int")
    #    print("Found circles")
    #    for (x, y, r) in circles:
    #        cv2.circle(image,(x,y),r,(64),4)
    #        cv2.rectangle(image,(x-5,y-5),(x+5,y+5),(255),-1)
      
    t1=time.time()
    print(str(t1-t0))
    # show the frame
    cv2.imshow("Frame", image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # lock camera settings
        if True:
            camera.shutter_speed = camera.exposure_speed
            camera.exposure_mode = 'off'
            g = camera.awb_gains
            camera.awb_mode = 'off'
            camera.awb_gains = g
    #if key == ord("c"):
    #    cChain = (cChain + 1) % 3
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    if key == ord("q"):
        break
cv2.destroyAllWindows()

