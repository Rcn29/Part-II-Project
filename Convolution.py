import sys
import os
os.chdir('/home/pi')
sys.path.append('')
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
from matplotlib import pyplot as plt
import glob

def extractNumberFrom(fileName):
    nr=0
    i=0
    
    while not fileName[i].isdigit() and i<len(fileName):
        i+=1
        
    while fileName[i].isdigit() and i<len(fileName):
        nr*=10
        nr+=int(fileName[i])
        i+=1
        
    return nr

index=0
bgr=glob.glob('/home/pi/Background.jpg')
background=cv2.imread(bgr[0])
grayground=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
images=glob.glob('/home/pi/FrameNumber*.jpg')
for fname in images:
    img=cv2.imread(fname)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=gray-grayground
    edges=cv2.Canny(img,50,120)
    index=extractNumberFrom(fname)
    cv2.imwrite('GrayFrameNumber'+str(index)+'.jpg',gray)
    cv2.imwrite('EdgeFrameNumber'+str(index)+'.jpg',edges)
    plt.subplot(121),plt.imshow(gray,cmap='gray')
    plt.title('Original grayscale image'),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap='gray')
    plt.title('Edge image'),plt.xticks([]),plt.yticks([])
    plt.show()