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
    
    
images=glob.glob('CalibrationSerban/Worked/Frame*.jpg')
print(images)
    
objpoints=None
imgpoints=None
w=1280#1920 #for the input image
h=720#1080
print(w,h)
force=False
try:
    with np.load("calib_serban.npz") as savedPoints:
        objpoints = savedPoints["obj"]
        imgpoints = savedPoints["img"]
        #w=1920
        #h=1080
except:
    pass

if force or (objpoints is None):
    objpoints = []
    imgpoints = []
    index=0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp=np.zeros((6*7,3),np.float32)
    objp[:,:2]=np.mgrid[0:7,0:6].T.reshape(-1,2)
    for fname in images:
        img=cv2.imread(fname)
        img=cv2.resize(img, (w,h))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print(gray)
        #cv2.imshow('gray',gray)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()
        ret, corners=cv2.findChessboardCorners(gray,(7,6),None)
        print("At "+str(extractNumberFrom(fname)))
        if ret == True:
            objpoints.append(objp)
            print(fname)
            print('index: '+str(index))
            cornersImg=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(cornersImg)
            index+=1
            img=cv2.drawChessboardCorners(img,(7,6),cornersImg,ret)
            #cv2.imshow('img', img)
            #cv2.imwrite('ChessboardWorkingInitial'+str(index)+'.jpg',img)
            cv2.imwrite('ChessboardWorkingDrawn'+str(index)+'.jpg',img)
            #cv2.waitKey(500)
            #cv2.destroyAllWindows()
    np.savez("calib_serban.npz", obj=objpoints, img=imgpoints)

#print(objpoints)
#print(imgpoints)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h),None,None)
# hack to remove translation? to make ROI work below
dist[0,2] = 0
dist[0,3] = 0
dist[0,4] = 0
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h), 1.0, (640,360))
print(roi)
x,y,wd,hd=roi
for fname in images:
    img=cv2.imread(fname)
    img=cv2.resize(img, (w,h))
    t0=time.time()
    dst=cv2.undistort(img,mtx,dist,None,newcameramtx)
    t1=time.time()
    dst=dst[y:y+hd,x:x+wd]
    t2=time.time()
    print("Undistort:"+str(t1-t0))
    print("Slice:"+str(t2-t1))
    number=extractNumberFrom(fname)
    cv2.imwrite('CalibrationSerban/Worked/'+str(number)+'Undistorted.jpg',dst)

print('Done')
cv2.destroyAllWindows() 