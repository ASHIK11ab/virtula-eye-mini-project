import cv2
import numpy as np
from time import time

camera = Camera()
cnt = 0

try:
  for frame in camera.capture():
    cnt += 1
    print(f"Frame: {cnt}")
except KeyboardInterrupt:
  print(f"Total frames: {cnt}")