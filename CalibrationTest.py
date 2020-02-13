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
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp=np.zeros((6*7,3),np.float32)
objp[:,:2]=np.mgrid[0:7,0:6].T.reshape(-1,2)
objpoints=[]
imgpoints=[]
images=glob.glob('/home/pi/WorkedCalibration/ChessboardWorkingInitial*.jpg')
print(images[0])
for fname in images:
    img=cv2.imread(fname)
    gray2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(gray2)
    #cv2.imshow('gray2',gray2)
    #cv2.waitKey(500)
    #cv2.destroyAllWindows()
    ret, corners=cv2.findChessboardCorners(gray2,(7,6),None)
    
    if ret == True:
        objpoints.append(objp)
        print(fname)
        print('index: '+str(index))
        corners2=cv2.cornerSubPix(gray2,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img2=img.copy()
        img2=cv2.drawChessboardCorners(img2,(7,6),corners2,ret)
        #cv2.imshow('img', img2)
        index+=1
        #cv2.imwrite('ChessboardWorkingInitial'+str(index)+'.jpg',img)
        #cv2.imwrite('ChessboardWorkingDrawn'+str(index)+'.jpg',img2)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()
        
        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray2.shape[::-1],None,None)
for fname in images:
    img=cv2.imread(fname)
    h,w=img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst=cv2.undistort(img,mtx,dist,None,newcameramtx)
    x,y,w,h=roi
    dst=dst[y:y+h,x:x+w]
    number=extractNumberFrom(fname)
    cv2.imwrite('ChessboardUndistorted'+str(number)+'.jpg',dst)

print('Done')
cv2.destroyAllWindows()