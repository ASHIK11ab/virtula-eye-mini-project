from Detector import *

CLASSES_FILE_PATH = "coco.names"
MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
TEST_IMAGE_PATH = "assets/test/2.jpg"
VIDEO_SRC = 0 # Webcam
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
MAX_OBJECTS = 20

detector = Detector()
detector.loadClasses(CLASSES_FILE_PATH)
detector.downloadModel(MODEL_URL)
detector.loadModel()
# detector.predictImage(TEST_IMAGE_PATH, MAX_OBJECTS, IOU_THRESHOLD, CONFIDENCE_THRESHOLD)

detector.predictVideo(VIDEO_SRC, MAX_OBJECTS, IOU_THRESHOLD, CONFIDENCE_THRESHOLD)