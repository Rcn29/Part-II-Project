import sys
#print(sys.path)
import os
#print(os.getcwd())
os.chdir('/home/pi')
#print(os.getcwd())
sys.path.append('')
import cv2
from picamera.array import PiRGBArray
import numpy as np

snapSpeed = 0.5
#NOTE: All of these images are lower resolution than input
scaleFactor = 4
prevImage = None # unmodified previous frame contents
motionHist = None # accumulated motion pixels
motionMask = None # mask of pixels that moved 
dilateMotion = np.ones((5,5), np.uint8)

def updateMotion(image):
    global motionHist
    global prevImage
    global motionMask
    motionDim = (int(image.shape[1] / scaleFactor), int(image.shape[0]/scaleFactor))
    image = cv2.resize(image, motionDim, cv2.INTER_LINEAR)
    if prevImage is not None:
        motionImg = cv2.absdiff( image, prevImage)
        motionImg = cv2.cvtColor(motionImg, cv2.COLOR_BGR2GRAY)
        motionImg = cv2.GaussianBlur(motionImg,(5,5),0)
        t = snapSpeed
        if motionHist is None:
            motionHist = motionImg.copy()
        #cv2.accumulateWeighted( motionImg, motionHist, t)
        motionHist = cv2.addWeighted( motionHist, 1.0 - t, motionImg, t, 0.0)
        motionHist = cv2.dilate(motionHist, dilateMotion)
        _, motionMask = cv2.threshold(motionHist, 15, 255, cv2.THRESH_BINARY)
    prevImage = image.copy()

def apply(image):
    global motionMask
    #image = cv2.absdiff( image, background)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #motion = cv2.absdiff(image, ( bgHist / 255).astype('uint8') )
    #ret, motionMask = cv2.threshold(motion, 50, 255, cv2.THRESH_BINARY)
    if motionMask is None:
        return image
    
    histImg = motionMask
    if (image.ndim == 3):
        histImg = cv2.cvtColor(motionMask, cv2.COLOR_GRAY2BGR)
    fullMask = cv2.resize(histImg, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
    #image = cv2.absdiff(image, bgHist)
    image = fullMask & image
    return image
