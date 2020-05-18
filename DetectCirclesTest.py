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

forceWidth = 960 // 1
forceHeight = 544 // 1

def ReadImgData(filename):
    image=cv2.imread(filename)
    image=cv2.resize(image, (forceWidth, forceHeight))
    return image#cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
t0=time.time()
folderName="NaturalLight/"
background = ReadImgData(folderName + "Background.ppm")
image = cv2.absdiff( ReadImgData(folderName + "Frame2.ppm"), background)
#backgroundSubtractor = cv2.createBackgroundSubtractorKNN()
# backgroundSubtractor.apply(background)
image = cv2.GaussianBlur(image,(3,3),0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
#image = cv2.adaptiveThreshold( image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
kernel = np.ones((3,3), np.uint8)
#image = cv2.morphologyEx( image, cv2.MORPH_OPEN, kernel, iterations = 1)

#gray=cv2.Canny(image,80,160)
#gray=abs(255-gray)
"""
circlesSrc = image
circles=cv2.HoughCircles(circlesSrc,cv2.HOUGH_GRADIENT, 1, 60,param1=550,param2=50,maxRadius=40)
output=circlesSrc.copy()
if circles is not None:
    circles=np.round(circles[0,:]).astype("int")
    print("Found circles")
    for (x, y, r) in circles:
        cv2.circle(output,(x,y),r,(0,255,0),4)
        cv2.rectangle(output,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
        """
output = image 
t1=time.time()
print(t1-t0)
cv2.imshow("output",output)
cv2.waitKey(0)
 
    