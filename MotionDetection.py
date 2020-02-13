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
#from Convolution import extractNumberFrom

def CalculatePixelDistance(p1,p2):
    return ((int(p1[0])-int(p2[0]))**2 + (int(p1[1])-int(p2[1]))**2 +
            (int(p1[2])-int(p2[2]))**2)

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

camera_width=1280
camera_height=1024
threshold=600

camera=PiCamera()
camera.resolution=(camera_width,camera_height)
#camera.framerate=1
rawCapture=PiRGBArray(camera,size=(camera_width,camera_height))
time.sleep(0.1)
camera.capture(rawCapture,format="bgr")
previous=rawCapture.array
grayPrev=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
rawCapture=PiRGBArray(camera,size=(camera_width,camera_height))
index=0
frames=glob.glob("EdgeFrameNumber*.jpg")
frames.sort()
print(frames)
for frame in frames:
    image=cv2.imread(frame)
    t0=time.time()
    #image=cv2.GaussianBlur(image,(5,5),0)
    #print("Gaussian took " + str(t1-t0) + "seconds")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    """
    for i in range(camera_height):
        for j in range(camera_width):
            distance=CalculatePixelDistance(previous[i][j],
                                            image[i][j])
            if(distance>threshold):
                toShow[i][j]=200
            else:
                toShow[i][j]=0
    """
    #toShow=(image-previous)**2
    grayShow=np.abs(gray-grayPrev)
    #minimumGray=np.amin(grayShow)
    #grayShow=grayShow+minimumGray
    #maximumGray=np.amax(grayShow)
    #grayShow=(grayShow/maximumGray) * 255
    #grayShow=np.maximum(100,grayShow)
    t1=time.time()
    index=extractNumberFrom(frame)
    print("Frame number "+str(index) + "Took " + str(t1-t0)+ "seconds")
    #cv2.imshow("Converted",toShow)
    cv2.imshow("ConvertedGray",grayShow)
    cv2.imwrite("MotionDetectedFrame"+str(index)+".jpg",grayShow)
    #previous=image.copy()
    grayPrev=gray.copy()
    key=cv2.waitKey(1)& 0xFF
    rawCapture.truncate(0)
