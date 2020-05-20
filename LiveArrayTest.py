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
import MotionMask

def stopCameraAutos(camera):
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g

inputW = 1280
inputH = 768
inputScale = int(inputW / 640)

camera = PiCamera()
camera.resolution = (inputW, inputH)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(inputW,inputH))
# allow the camera to warmup
#camera.start_preview() #DO NOT DO THIS
camera.capture(rawCapture, format="bgr", use_video_port=True)
rawCapture.truncate(0)
#time.sleep(3.0)
#stopCameraAutos(camera)

"""
cv2.namedWindow("Frame")
cannyMin = 80
cannyMax = 140
def trackMin(v) :
    global cannyMin
    cannyMin = v
def trackMax(v) :
    global cannyMax
    cannyMax = v
#cv2.createTrackbar("Canny Min", "Frame" , cannyMin, 500, trackMin)
#cv2.createTrackbar("Canny Max", "Frame" , cannyMax, 500, trackMax)
"""

minT = [1,2,3]
maxT = [1,2,3]
flexMin = 0.98
flexMax = 1.02
adjusting = True # picking threshold values
cycle = 0 # cycle through displayed images

def subimg(img, sz, px,py):
    szh=int(sz * 0.5)
    x1 = px-szh
    x2 = px+szh
    y1 = py-szh
    y2 = py+szh
    return img[y1:y1+sz,x1:x1+sz,:]

def writesubimg(src, dst, sz, px,py):
    szh=int(sz * 0.5)
    x1 = px-szh
    x2 = px+szh
    y1 = py-szh
    y2 = py+szh
    dst[y1:y1+sz,x1:x1+sz,:] &= src

dilateCanny = np.ones((2,2), np.uint8)
erodeMask = np.ones((2 + 3 * inputScale,2 + 3 * inputScale), np.uint8)

def readThreshValues(sz, px, py, img, imgHSV, show):
    global minT
    global maxT
    global midH
    
    sub = subimg(imgHSV, sz,px,py)
    sub = cv2.GaussianBlur(sub, (1 + 2*inputScale, 1 + 2*inputScale), 0)
    # create Hue & Value edges
    cnyMin = 80
    cnyMax = 120
    cny = cv2.Canny(sub[:,:,0], cnyMin, cnyMax, 7)
    cny = cv2.bitwise_or(cny, cv2.Canny(sub[:,:,1], cnyMin, cnyMax, 7))
    cny = cv2.bitwise_or(cny, cv2.Canny(sub[:,:,2], cnyMin, cnyMax, 7))
    cny = cv2.dilate(cny, dilateCanny)
    #cny = cv2.GaussianBlur(cny, (3,3), 0)
    #cv2.copyMakeBorder(cny,1,1,1,1, cv2.BORDER_ISOLATED | cv2.BORDER_CONSTANT, cny,(255))
    if show:
        cv2.imshow("Sub", sub[:,:,2])
        cv2.imshow("Cny", cny)
    # fill area edges to get inverted hand outline 
    cv2.floodFill(cny, None,(0,0),(64))
    cv2.floodFill(cny, None,(sz -1,sz -1),(64))
    cv2.floodFill(cny, None,(sz -1,0),(64))
    cv2.floodFill(cny, None,(0,sz -1),(64))
    # select only floodfilled pixels and invert to get actual hand outline
    cny = cv2.compare(cny,(64), cv2.CMP_EQ)
    cny = cny ^ (255)
    cny = cv2.erode(cny, erodeMask)
    #cny = cv2.dilate(cny, erodeMask)
    if show:
        cv2.imshow("Flood", cny)
    
    # preview over image
    bla = cv2.cvtColor(cny, cv2.COLOR_GRAY2BGR)
    writesubimg(bla, img, sz, px,py)
    
    # use masked pixels to guess hand colors
    for i in range(3):
        minV,maxV,_,_ = cv2.minMaxLoc( sub[:,:,i], cny)
        minT[i] = minV# * flexMin
        maxT[i] = maxV# * flexMax
        
    dist=maxT[0]-minT[0]
    if dist>90.0:#Wraps around
        dist = dist - 180
    midH = minT[0] + dist * 0.5
    if midH < 0:
        midH += 180

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    t0=time.time()
    image = np.array(frame.array)
    #image = image[ 250:750, 700:1400,:]
    image = cv2.flip(image,-1)
    #image =cv2.GaussianBlur(image, (3,3), 0)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #print("Capture " + str(time.time() - t0))
    
    # mask moving pixles
    MotionMask.updateMotion(image)
    
    # threshold adjust square
    cx = int(inputW * 0.5)
    cy = int(inputH * 0.5)
    sz = 40 * inputScale
    szh=int(sz/2)
    x1 = cx-szh
    x2 = cx+szh
    y1 = cy-szh
    y2 = cy+szh
    if adjusting:
        """
        for i in range(3):
            imageH = imageHSV[y1:y1+sz,x1:x1+sz, i]
            minV,maxV,_,_ = cv2.minMaxLoc( imageH)
            minT[i] = minV * flexMin
            maxT[i] = maxV * flexMax
        """
        readThreshValues(sz, cx,cy, image, imageHSV, show=True) 
    
    # find individual motion islands
    islands, _ = cv2.findContours(MotionMask.motionMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for ct in islands: 
        ct *= 4    
    
    #find contours inside each island
    for island in islands:
        x,y,w,h = cv2.boundingRect(island)
        cv2.rectangle(image, (x,y),(x+w,y+h), (200,0,0))
        
        subHSV = imageHSV[y:y+h,x:x+w,:]
        # replace H channel with a "distance from hand hue" value
        hd = cv2.absdiff( subHSV[:,:,0], (midH))
        hd = cv2.min(hd, (180) - hd)
        subHSV[:,:,0] = hd
        subThresh = cv2.inRange(subHSV, (0, minT[1], minT[2]), (20, maxT[1], maxT[2]))
        subThresh = cv2.GaussianBlur(subThresh, (3,3), 0)
        
        #subMask = cv2.cvtColor(subThresh, cv2.COLOR_GRAY2BGR)
        #cv2.imshow("Smask", subMask)
        #image[y:y+h,x:x+w,:] = subMask
        
        conts, _ = cv2.findContours(subThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(x,y))
        cv2.drawContours(image, conts, -1, (0, 0, 200), 2)
        if len(conts):
            allConts = np.concatenate(conts)
            #cv2.drawContours(image, [allConts], -1, (0, 0, 200), 2)
            hull = cv2.convexHull(allConts, returnPoints=False)
            if len(hull):
                hullPts = allConts[hull]
                cv2.drawContours(image, hullPts, -1, (250, 0, 200), 4)        
                defects = cv2.convexityDefects(allConts, hull)
                if defects is not None:
                    for defect in defects:
                        st,end,far,depth = defect[0]
                        #far = defect
                        #print(allConts[far])
                        if depth > 5:
                            cv2.circle(image, tuple(allConts[far][0]), 2, (200,200,200), 2)
    
    # draw masked image
    if False:
        # replace H channel with a "distance from hand hue" value
        hd = cv2.absdiff( imageHSV[:,:,0], (midH))
        hd = cv2.min(hd, (180) - hd)
        imageHSV[:,:,0] = hd
        imageThresh = cv2.inRange(imageHSV, (0, minT[1], minT[2]), (20, maxT[1], maxT[2]))
        #imageThresh = cv2.GaussianBlur(imageThresh, (5,5), 0)
        imageThresh = MotionMask.apply(imageThresh)
        imageThresh = cv2.cvtColor(imageThresh, cv2.COLOR_GRAY2BGR)
        imageMasked = cv2.addWeighted( image, 1, imageThresh, 0.5, 0.0)
    
    #print("Mask " + str(time.time() - t0))
    
    image = cv2.drawContours(image, islands, -1, (0, 200,0), 1)
    
    if adjusting:
        cv2.rectangle(image, (x1,y1),(x2,y2), (0,200,0), 2)
    cv2.putText( image,
     "Hue " + str(minT) + " " + str(maxT), (40, 40),
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     0.25 * inputScale, #font size
     (209, 80, 0), #font color
     2)
    cv2.putText( image,
     "Mid H " + str(midH), (40, 90),
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     0.25 * inputScale, #font size
     (209, 80, 0), #font color
     2)
    
    #image=cv2.Canny(image,50,120)
    # show the frame
    #cv2.imshow("Frame", imageHSV[:,:,2])
    if (cycle == 0):
        cv2.imshow("Frame", image)
    #if (cycle == 1):
    #    cv2.imshow("Frame", imageThresh)
    #if (cycle == 2):
    #    cv2.imshow("Frame", imageMasked)
    if (cycle == 3):
        cv2.imshow("Frame", imageHSV[:,:,0])
    if (cycle == 4):
        cv2.imshow("Frame", imageHSV[:,:,1])
    if (cycle == 5):
        cv2.imshow("Frame", imageHSV[:,:,2])
    if (cycle == 6):
        pass
    
    t1=time.time()
    print(str(t1-t0))
    
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    if key == ord("s"):
        adjusting = not adjusting
        stopCameraAutos(camera)
    if key == ord("m"):    
        cycle = (cycle + 1) % 6
    if key == ord("q"):
        break
cv2.destroyAllWindows()