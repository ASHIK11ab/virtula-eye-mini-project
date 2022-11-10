import cv2, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
from Util import object_in_walk_frame

np.random.seed(123)

class Detector:
  def __init__(self):
    self.classList = None
    self.colorList = None
    self.cacheDir = None
    self.modelName = None
    self.model = None

  def loadClasses(self, classesFilePath):
    with open(classesFilePath, 'r') as file:
      self.classList = file.read().splitlines()

    self.colorList = np.random.uniform(low = 0, high = 255, size=(len(self.classList), 3))


  def downloadModel(self, modelURL):
    filename = modelURL.rsplit("/", 1)[1]
    self.modelName = filename.split(".")[0]

    self.cacheDir = "./pretrained"
    os.makedirs(self.cacheDir, exist_ok=True)

    get_file(fname=filename, origin=modelURL, cache_dir=self.cacheDir,
              cache_subdir="checkpoints", extract=True)

  
  def loadModel(self):
    print("Loading model " + self.modelName)
    
    tf.keras.backend.clear_session()
    self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))

    print("\nmodel loading successfull")


  def createBoundingBox(self, image, maxObjects, iou_threshold, confidenceThreshold):
    inputTensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
    inputTensor = inputTensor[tf.newaxis, ...]

    detections = self.model(inputTensor)
    bboxs = detections['detection_boxes'][0].numpy()
    classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
    classScores = detections['detection_scores'][0].numpy()

    imH, imW, imC = image.shape

    mid = imW // 2
    leftLineX = mid - 200
    rightLineX = mid + 200

    # Filter detected objects and cancel noise
    bboxIds = tf.image.non_max_suppression(bboxs, classScores, max_output_size=maxObjects, iou_threshold=iou_threshold, score_threshold=confidenceThreshold)

    if len(bboxIds) != 0:
      for i in bboxIds:
        box = tuple(bboxs[i].tolist())
        classConfidence = round(100 * classScores[i])
        classIndex = classIndexes[i]

        classLabelText = self.classList[classIndex]
        classColor = self.colorList[classIndex]

        displayText = f"{classLabelText}: {classConfidence}%"

        ymin, xmin, ymax, xmax = box
        xmin, xmax, ymin, ymax = int(xmin * imW), int(xmax * imW), int(ymin * imH), int(ymax * imH)
        
        print(xmin, ymin, xmax, ymax)
        print(leftLineX, rightLineX)

        # left line
        cv2.line(image, (leftLineX, 0), (leftLineX, imH), (255, 0, 0), thickness=2)
        # right line
        cv2.line(image, (rightLineX, 0), (rightLineX, imH), (255, 0, 0), thickness=2)
        
        # ignore objects which are not in walk frame.
        if object_in_walk_frame(leftLineX, rightLineX, (xmin, ymin), (xmax, ymax)):
          cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2)
          cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color=classColor, thickness=2)
    
    return image


  def predictVideo(self, src, maxObjects, iou_threshold, confidenceThreshold):
    video = cv2.VideoCapture(src)
  
    while video.isOpened():
      success, frame = video.read()

      if success:
        bbox = self.createBoundingBox(frame, maxObjects, iou_threshold, confidenceThreshold)
        cv2.imshow("Result", bbox)
        cv2.waitKey(10)
      else:
        break
    video.release()
    cv2.destroyAllWindows()