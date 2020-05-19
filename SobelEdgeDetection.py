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

images=glob.glob('FrameNumber18.jpg')
img=cv2.imread(images[0])
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grad_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
grad_y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
abs_grad_x=cv2.convertScaleAbs(grad_x)
abs_grad_y=cv2.convertScaleAbs(grad_y)
grad=cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
ret,grad=cv2.threshold(grad,70,255,cv2.THRESH_BINARY)
canny=cv2.Canny(gray,100,150)
cv2.imshow("Sobel",grad)
cv2.waitKey(2000)
cv2.imwrite("SobelExample.jpg",grad)
cv2.destroyAllWindows()
cv2.imshow("Canny",canny)
cv2.imwrite("CannyExample.jpg",canny)
cv2.waitKey(0)