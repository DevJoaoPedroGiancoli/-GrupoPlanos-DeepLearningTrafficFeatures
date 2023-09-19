import cv2, time, os, tensorflow as tf
import numpy as np
from Model import *
from Detector import *
from tensorflow.python.keras.utils.data_utils import get_file


class BoundigBox:
    def __init__(self, detector, model):
        self.detector = detector
        self.model = model
        
    def createBoundigBox(self, image):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]
        
        detections = self.model(inputTensor)
        
        bboxs= detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()
        
        imH, imW, imC = image.shape
        
        if len(bboxs) != 0:
            for i in range(0, len(bboxs)):
                for i in range(0, len(bboxs)):
                    bbox = tuple(bbox[i].tolist())
                    classConfidence = round(100*classScores[i])
                    classIndex = classIndexes[i]
                    
                    classLabelText = self.classesList[classIndex]
                    classColor = self.colorList[classIndex]
                    
                    displayText = '{}: {}%'.format(classLabelText, classConfidence)
                    
                    ymin, xmin, ymax, xmax = bbox
                    
                    ymin, xmin, ymax, xmax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                    
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color=classColor, thickness=1)