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

backSub=cv2.createBackgroundSubtractorKNN()
backgroundFile=glob.glob("AcasaFrameNumber20.jpg")
imageFile=glob.glob("AcasaFrameNumber*.jpg")
background=cv2.imread(backgroundFile[0])
bla=backSub.apply(background)

for i in range(len(imageFile)-1):
    for j in range(1,len(imageFile)-i-1):
        if(extractNumberFrom(imageFile[j+i])<extractNumberFrom(imageFile[i])):
            aux=imageFile[j+i]
            imageFile[j+i]=imageFile[i]
            imageFile[i]=aux

for file in imageFile:
    img=cv2.imread(file)
    print(file)
    t0=time.time()
    fGround=backSub.apply(img)
    t1=time.time()
    print(t1-t0)
    cv2.imshow('Foreground',fGround)
    cv2.waitKey(0)
    