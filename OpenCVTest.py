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

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp=np.zeros((5*5,3),np.float32)
objp[:,:2]=np.mgrid[0:5,0:5].T.reshape(-1,2)
objpoints=[]
imgpoints=[]
images=glob.glob('/home/pi/Chessboard*.jpg')
for fname in images:
    img=cv2.imread(fname)
    cv2.imshow('img',img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, corners=cv2.findChessboardCorners(gray,(5,5),None)
    
    if ret == True:
        objpoints.append(objp)
        
        corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        
        img=cv2.drawChessboardCorners(img,(5,5),corners2,ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        
cv2.destroyAllWindows()