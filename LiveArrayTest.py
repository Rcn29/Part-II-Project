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
from pathlib import Path
from math import sqrt
import matplotlib.pyplot as plt
import TestHelper as tst

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

#bgImage = np.zeros((inputH,inputW,3), np.uint8)
minT = [1,2,3]
maxT = [1,2,3]
flexMin = 0.98
flexMax = 1.02
adjusting = True # picking threshold values
cycle = 0 # cycle through displayed images
lastMovingIsland=None
testToggle = False
lastFingers = []
goodFingers = []
wholeImage=False

crtFrame=0
doPrint=False
folderName='TestNumber'
try:
    with np.load("testnumber.npz") as numberFile:
        testNumber=numberFile["nr"]
except:
    testNumber=6
fakeFrames = None


def fingerDists(p1,p2,p3):
    p21 = p2-p1
    baseD = np.linalg.norm(p21)
    #distance from p3 to line [p1,p2]
    farD = np.linalg.norm(np.cross(p21, p1-p3)) / baseD
    return farD,baseD

def findFingers(conts, image, roi):
    fingers = []
    bArea = 0
    if len(conts) > 10:
        return fingers, bArea
    
    for cont in conts:
        cv2.drawContours(image, [cont], -1, (0, 0, 200), 1)
        hull = cv2.convexHull(cont, returnPoints=False)
        if len(hull) == 0:
            continue
        hull = np.hstack(hull)
        hullPts = cont[hull]
        hullArea = cv2.contourArea(hullPts)
        if hullArea > bArea:
            bCont = cont
            bHull = hull
            bHullPts = hullPts
            bArea = hullArea
            
    if bArea is not 0:
        hullPts = bHullPts
        cont = bCont
        hull = bHull
        cv2.drawContours(image, [hullPts], -1, (250, 0, 200), 4)
        defects = cv2.convexityDefects(cont, hull)
        if defects is not None:
            maybe = []
            lastEnd = None
            #print(len(defects))
            for defect in defects:
                st,end,far,depth = defect[0]
                #thisEnd = conts[st][0]
                #if (lastEnd is not None and np.linalg.norm(thisEnd - lastEnd) < 15 ):
                #    fingers.append( tuple(thisEnd))
                #lastEnd = thisEnd
                # work out actual depth
                p1 = cont[st][0]
                p2 = cont[end][0]
                p3 = cont[far][0]
                farD,baseD = fingerDists(p1,p2,p3)
                if farD > 0.15*baseD and farD > 5:
                    if st not in maybe:
                        maybe.append(st)
                    if end not in maybe:
                        maybe.append(end)
                    cv2.circle(image, tuple(p3), 2, (200,200,200), 2)
                    cv2.circle(image, tuple(p1), 2, (50,200,50), 2)
                    cv2.circle(image, tuple(p2), 2, (50,200,50), 2)
            # merge close candidates
            mergeDist = 8 * (sqrt(bArea) / 50)
            i = 0
            num = len(maybe)
            m = maybe
            while i < num:
                j = i+1
                p = cont[m[i]][0]
                # skip edge points
                if min(abs(p[0] - roi[0]), abs(roi[1] - p[0])) < 5 or \
                    min(abs(p[1] - roi[2]), abs(roi[3] - p[1])) < 5:
                    i+=1
                    continue
                # remove other close pts
                while j < num:
                    p2 = cont[m[j]][0]
                    if np.linalg.norm(p - p2) < mergeDist:
                        cont[m[j]][0] = cont[m[num-1]][0]
                        num -= 1
                    else:
                        j+=1
                i+=1
                fingers.append(p)
            #for midx in maybe:
            #    fingers.append( tuple(conts[midx][0]))
    return fingers, bArea

def subimg(img, roi):
    return img[roi[2]:roi[3], roi[0]:roi[1],:]

def writesubimg(src, dst, roi):
    dst[roi[2]:roi[3], roi[0]:roi[1],:] &= src

dilateCanny = np.ones((5,5), np.uint8)
erodeMask = np.ones((0 + 1 * inputScale,0 + 1 * inputScale), np.uint8)

def readThreshValues(sz, px, py, img, imgHSV, show):
    global minT
    global maxT
    global midH
    
    szh=int(sz * 0.5)
    roi = (px-szh, px+szh,py-szh,py+szh)
    sub = subimg(imgHSV, roi)
    sub = cv2.GaussianBlur(sub, (1 + 2*inputScale, 1 + 2*inputScale), 0)
    # create Hue & Value edges
    cnyMin = 80
    cnyMax = 120
    cny = cv2.Canny(sub[:,:,0], cnyMin, cnyMax, 7)
    cny = cv2.bitwise_or(cny, cv2.Canny(sub[:,:,1], cnyMin, cnyMax, 7))
    cny = cv2.bitwise_or(cny, cv2.Canny(sub[:,:,2], cnyMin, cnyMax, 7))
    if show:
        cv2.imshow("Cny", cny)
    cny = cv2.morphologyEx(cny, cv2.MORPH_CLOSE, dilateCanny)  #cv2.dilate(cny, dilateCanny) #dilate to make contours more watertight
    if show:
        cv2.imshow("Cny Close", cny)
    #cny = cv2.GaussianBlur(cny, (3,3), 0)
    #cv2.copyMakeBorder(cny,1,1,1,1, cv2.BORDER_ISOLATED | cv2.BORDER_CONSTANT, cny,(255))
    #if show:
    #    cv2.imshow("Sub", sub[:,:,2])
    #    cv2.imshow("Cny", cny)
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
    #if show:
    #    cv2.imshow("Flood", cny)
    
    # preview over image
    bla = cv2.cvtColor(cny, cv2.COLOR_GRAY2BGR)
    writesubimg(bla, img, roi)
    
    # use masked pixels to guess hand colors
    for i in range(3):
        minV,maxV,_,_ = cv2.minMaxLoc( sub[:,:,i], cny)
        minT[i] = int(minV)# * flexMin
        maxT[i] = int(maxV)# * flexMax
    if True:
        # use masked HSV histogram to guess hand color
        #channels = [0 ,1,2]
        #histSize = [180, 50, 50]
        #ranges = [0,179,0,255,0,255]
        #hist = cv2.calcHist( [sub],[0,1,2], cny, histSize, ranges)
        histHue = cv2.calcHist( [sub],[0], cny, [180], [0,179])
        mostCommonHue = np.argmax(histHue) # pick most common hue
        midH = int(mostCommonHue)
        # render histogram
        def drawHist(hist, x, y):
            for i in range( len(hist)):
                cv2.rectangle(img, (x + i * 2, y), (x + 2 + i*2, y + hist[i]), (200, 200,200), -1)
        
        drawHist(histHue, 10, 95)
        #for i in range(180):
        #    cv2.rectangle(img, (10 + i * 2, 95), (12 + i*2, 95 + histHue[i]), (200, 200,200), -1)
        histS = cv2.calcHist( [sub],[1], cny, [128], [0,255])
        drawHist(histS, 400, 95)
        cv2.rectangle(img, (400 + minT[1], 85), (400 + maxT[1], 95), (0, 128,0), -1)
        histV = cv2.calcHist( [sub],[2], cny, [128], [0,255])
        drawHist(histV, 610, 95)
        cv2.rectangle(img, (610 + minT[2], 85), (610 + maxT[2], 95), (0, 128,0), -1)
    else:
        # pick middle HUE between extremes
        dist=maxT[0]-minT[0]
        if dist > 90.0:
            dist = dist - 180
        midH = minT[0] + dist * 0.5
        if midH < 0:
            midH += 180
        
    minT[2] = 0
    maxT[2] = 255
    # return ROI
    return roi

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    t0=time.time()
    
    if fakeFrames is None:
        image = np.array(frame.array)
        #image = image[ 250:750, 700:1400,:]
        image = cv2.flip(image,-1)
    else:
        image = fakeFrames.next().copy()
        
    originalImage=image.copy()
    #image =cv2.GaussianBlur(image, (3,3), 0)
    #print("Capture " + str(time.time() - t0))
    
    # mask moving pixles
    MotionMask.updateMotion(image)
    # keep good fingers around
    """
    for finger in lastFingers:
        s = 5
        sc = 1 / MotionMask.scaleFactor
        #print(finger)
        st = (int(finger[0]*sc)-s, int(finger[1]*sc)-s)
        end= (int(finger[0]*sc)+s, int(finger[1]*sc)+s)
        cv2.rectangle(MotionMask.motionMask, st, end,(255), 5)
    """
    # failed bg test
    if False and not adjusting:
        mask = cv2.resize(MotionMask.motionMask, (inputW, inputH), cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        bgImage = (image & ~mask) + (bgImage & mask)

    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # threshold adjust square
    cx = int(inputW * 0.5)
    cy = int(inputH * 0.5)
    sz = 40 * inputScale
    adjustROI = None
    if adjusting:
        adjustROI = readThreshValues(sz, cx,cy, image, imageHSV, show=True) 
    
    # find individual motion islands
    islands, _ = cv2.findContours(MotionMask.motionMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for ct in islands: 
        ct *= 4    
    
    if len(islands):
        lastMovingIsland = None
    elif lastMovingIsland is not None:
        islands.append(lastMovingIsland)
    
    #find contours inside each island
    for island in islands:        
        if lastMovingIsland is None:
            lastMovingIsland = island
        if adjusting:
            continue
        x,y,w,h = cv2.boundingRect(island)
        cv2.rectangle(image, (x,y),(x+w,y+h), (200,0,0))
        if (w*h>int(inputW*inputH*0.5)):
            continue
        #todo: if rectangle is larger than 320x180, scale it down
        if not wholeImage:
            subHSV = imageHSV[y:y+h,x:x+w,:]
            #
            if False: # failed bg removal test
                subHSVBg = bgImage[y:y+h,x:x+w,:]
                subHSVBg = cv2.cvtColor(subHSVBg, cv2.COLOR_BGR2HSV)
                # select unchanged pixels and flip the mask
                bgThresh = 255 - cv2.inRange(cv2.absdiff(subHSVBg, subHSV),(0,0,0), (255, 30,30))
                morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                bgThresh = cv2.erode(bgThresh, morphKernel)
                image[y:y+h,x:x+w,:] &= cv2.cvtColor(bgThresh, cv2.COLOR_GRAY2BGR)
            
            # replace H channel with a "distance from hand hue" value
            hd = cv2.absdiff( subHSV[:,:,0], (midH))
            hd = cv2.min(hd, (180) - hd)
            subHSV[:,:,0] = hd
            subThresh = cv2.inRange(subHSV, (0, minT[1], minT[2]), (10, maxT[1], maxT[2]))
            #subThresh = cv2.GaussianBlur(subThresh, (3,3), 0)
            
            cntSrc = subThresh
            
            if False:
                subMasked = subHSV & cv2.cvtColor(subThresh, cv2.COLOR_GRAY2BGR) 
                c1 = 60
                c2 = 140
                subcny = cv2.Canny(subMasked[:,:,0], c1, c2)
                subcny = subcny | cv2.Canny(subMasked[:,:,1], c1, c2)
                subcny = subcny | cv2.Canny(subMasked[:,:,2], c1, c2)
                cntSrc = subcny
                
            #
            #cv2.imshow("Smask", subMask)
            #image[y:y+h,x:x+w,:] = subMask
            morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            cntSrc = cv2.morphologyEx(cntSrc, cv2.MORPH_CLOSE, morphKernel)
            
            roi = (x, x+w, y, y+h)
            
            subMask = cv2.cvtColor(cntSrc, cv2.COLOR_GRAY2BGR)
            writesubimg( subMask, image, roi)        
        conts, _ = cv2.findContours(cntSrc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS, offset=(x,y))
        #cv2.drawContours(image, conts, -1, (0, 0, 200), 2)
        numConts = len(conts)
        if numConts:
            fingers, handArea = findFingers(conts, image, roi)
            #lastFingers
            # compare new finger list with old one
            # and keep only close enough fingers
            # use hand area to adjust movement sensitivity
            #NOTE: assumes max speed of 1 hand per sec @0.1 FPS
            maxMove = max( 6, 1.0 * sqrt(handArea))
            if False: # keep only stable fingers test
                for new in fingers:
                    for old in lastFingers:
                        if np.linalg.norm(new - old) < maxMove:
                            goodFingers.append(new) # found good candidate
                            break
            lastFingers = fingers
            if len(fingers):
                #fCenter, fR = cv2.minEnclosingCircle( np.array(goodFingers))
                #cv2.circle(image, tuple(fCenter), int(fR), (255,0,255),1)
                for finger in fingers:
                    cv2.circle(image, tuple(finger), 5, (0,0,255), 2)
    
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
    
    if not adjusting:
        image = cv2.drawContours(image, islands, -1, (0, 200,0), 1)
    
    # picking square
    if adjustROI:
        x1,x2,y1,y2 = adjustROI;
        cv2.rectangle(image, (x1,y1),(x2,y2), (0,200,0), 1)
        cv2.rectangle(image, (x1,y1),(x1+2,y1+2), (0,0,200), 4)
        cv2.rectangle(image, (x2-2,y2-2),(x2,y2), (0,0,200), 4)
        cv2.rectangle(image, (x1,y2-2),(x1+2,y2), (0,0,200), 4)
        cv2.rectangle(image, (x2-2,y1),(x2,y1+2), (0,0,200), 4)        
        
    cv2.rectangle(image, (40, 75),(250, 70 + inputScale * 13), (200,200,200), -1)
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
    #    cv2.imshow("Frame", bgImage)
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
    frameTime = t1-t0
    print(str(frameTime))
    if fakeFrames and not adjusting:
        fakeFrames.frameTimes[fakeFrames.li] = frameTime
        print(" F: " + str(fakeFrames.li))        
        cv2.imwrite(folderName+str(fakeFrames.testNum)+"/Frame"+str(fakeFrames.li)+".jpg",image)
        if fakeFrames.i == 10 and fakeFrames.li == 11:
            np.savetxt(folderName+str(fakeFrames.testNum)+"/_times.txt", fakeFrames.frameTimes)
            print("Wrote " + str(fakeFrames.testNum))
            fakeFrames = tst.loadTest(fakeFrames.testNum + 1)
            adjusting = True
            #cv2.waitKey(0)
    
    if doPrint and adjusting==False and crtFrame<=40:
        cv2.imwrite(folderName+str(testNumber)+"/Frame"+str(crtFrame)+".ppm",originalImage)
        print("Wrote frame number " + str(crtFrame) + " to path" + folderName+str(testNumber))
        crtFrame+=1
        if crtFrame>40:
            doPrint=False
    
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    if key == ord("s") or (fakeFrames and fakeFrames.i == -1):
        adjusting = not adjusting
        stopCameraAutos(camera)
        if doPrint and adjusting==False:
            testNumber+=1
            Path(folderName+str(testNumber)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(folderName+str(testNumber)+"/HandSample.ppm",originalImage)
            crtFrame=0
            np.savez("testnumber.npz",nr=testNumber)
        if fakeFrames is not None:
            if not adjusting:
                fakeFrames.i = 0
            if adjusting:
                fakeFrames.i = -1
    if key == ord("m"):    
        cycle = (cycle + 1) % 6
    if key == ord("t"):
        testToggle = not testToggle
    if key == ord("r"):
        wholeImage= not wholeImage
    if wholeImage:
        subHSV = imageHSV.copy()
        flubHSV=imageHSV.copy()
        # replace H channel with a "distance from hand hue" value
        hd = cv2.absdiff( subHSV[:,:,0], (midH))
        hd = cv2.min(hd, (180) - hd)
        subHSV[:,:,0] = hd
        subThresh = cv2.inRange(subHSV, (0, minT[1], minT[2]), (10, maxT[1], maxT[2]))
        #subThresh = cv2.GaussianBlur(subThresh, (3,3), 0)

        cntSrc = subThresh
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        cntSrc = cv2.morphologyEx(cntSrc, cv2.MORPH_CLOSE, morphKernel)

        roi = (0, inputW, 0, inputH)

        subMask = cv2.cvtColor(cntSrc, cv2.COLOR_GRAY2BGR)
        writesubimg( subMask, flubHSV, roi)
        cv2.imshow("ColorMasked",flubHSV)
    if key == ord("q"):
        break
    if key == ord("p"):
        doPrint=True
    # saved tests
    keyNum = 1 + key - ord("1")
    if keyNum > 0 and keyNum < 10:
        fakeFrames = tst.loadTest( keyNum)
        adjusting = True
    
cv2.destroyAllWindows()