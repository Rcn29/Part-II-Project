import os.path

import sys
os.chdir('/home/pi')
sys.path.append('')
import cv2

folder = "TestNumber"

class Test:
    testNum = 0
    i = -1
    li = -1
    change = 1
    hand = None
    frames = None
    frameTimes = None
    
    def __init__(self, testNum, hand, frames):
        self.testNum = testNum
        self.hand = hand
        self.frames = frames
        self.frameTimes = [0.0] * len(frames) #np.zeros()
    
    def next(self):
        if self.i == -1:
            return self.hand
        i = self.i
        frame = self.frames[i]
        i += self.change
        if i >= len(self.frames):
            i -= 2
            self.change = -1
        if i < 0:
            i += 2
            self.change = 1
        if i<10 and self.change==-1:
            i+=1
            self.change = 1
        self.li = self.i
        self.i = i
        return frame
    

def loadTest(num):
    num = int(num)
    base = folder + str(num)
    samplePath = os.path.join(base, "HandSample.ppm")
    if not os.path.isfile(samplePath):
        return None
    
    handSample = cv2.imread(samplePath)
    frames = []
    i = 0
    framePath = os.path.join(base, "Frame" + str(i) + ".ppm")
    while os.path.isfile(framePath):
        frames.append(cv2.imread(framePath))
        i+=1
        framePath = os.path.join(base, "Frame" + str(i) + ".ppm")
    return Test(num, handSample, frames)

    