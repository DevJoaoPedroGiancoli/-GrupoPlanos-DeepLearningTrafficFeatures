import cv2, time, os, tensorflow as tf
import numpy as np
from Model import *
from BoundigBox import *
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self, model, boundigbox):
        self.model = model
        self.boundigbox = boundigbox
    
    def predictImage(self, imagePath):
        image = cv2.imread(imagePath)
        
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()