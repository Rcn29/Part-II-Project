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
import picamera
import time
import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib

print(str(np.version))
print(cv2.__version__)
print(matplotlib.__version__)