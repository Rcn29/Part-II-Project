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

#Command used :
#raspistill -o Chessboard1.jpg -hf -vf -w 1920 -h 1080 -t 5000

#Input list
toContrast=glob.glob("Chessboard4.jpg")

for fname in toContrast:
    img=cv2.imread(fname)
    #Grayscale image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Contrast image
    gray2=np.array(gray,np.uint8)
    for i in range(1080):
        for j in range(1920):
            if gray[i][j]>=100:
                gray2[i][j]=230
            else:
                gray2[i][j]=10
    #Overwrite original file (Or change fname to not overwrite)
    cv2.imshow('Contrasted image',gray2)
    cv2.waitKey(500)
    cv2.imwrite('Contrasted'+fname,gray2)
    
cv2.destroyAllWindows()
    