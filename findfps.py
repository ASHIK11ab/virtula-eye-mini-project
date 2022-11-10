import cv2
import numpy as np
from time import time

class Camera:
  def __init__(self):
    self.video = None

  def capture(self):
    video = cv2.VideoCapture(0)

    while video.isOpened():
      ret, frame = video.read()

      if ret:
        yield frame
      else:
        yield None
    yield None

camera = Camera()

cnt = 0
try:
  for frame in camera.capture():
    cnt += 1
    print(f"Frame: {cnt}")
except KeyboardInterrupt:
  print(f"Total frames: {cnt}")