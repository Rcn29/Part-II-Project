from picamera import PiCamera, Color
from time import sleep
camera=PiCamera()
camera.rotation = 180
camera.color_effects=(128,128)
camera.start_preview()
print(camera.resolution)
sleep(5)
camera.stop_preview()