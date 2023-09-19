import cv2, time, os, tensorflow as tf
import numpy as np
<<<<<<< HEAD
from Model import *
from BoundigBox import *
=======

>>>>>>> 821456f3be6b555845362a2e0c20c25eb026742b
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
<<<<<<< HEAD
    def __init__(self, model, boundigbox):
        self.model = model
        self.boundigbox = boundigbox
    
    def predictImage(self, imagePath):
        image = cv2.imread(imagePath)
        
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
=======
    def __init__(self) -> None:
        pass
    
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            print(f"esse eh o formato de classesFilePath: {classesFilePath} \n")
            self.classesList = f.read().splitlines()
            print(f'esse eh o formato de classesList: {self.classesList} \n')
            
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))
            
            print(len(self.classesList), len(self.colorList))
            
    def downloadModel(self, modelURL):
        
        fileName = os.path.basename(modelURL)
        
        print(f"Esse eh o formato de fileName: {fileName} \n")
        
        self.modelName = fileName[:fileName.index('.')]
        
        print(f"Esse eh o formato de modelName: {self.modelName} \n")
        
        
        #criando o diretorio
        self.cacheDir = r"CodingBug\pretrained_models"
        os.makedirs(self.cacheDir, exist_ok = True)
        
        get_file(fname = fileName,
                 origin=modelURL, cache_dir=self.cacheDir,
                 cache_subdir="checkpoints", extract=True)
        
    def loadModel(self):
        print(f"Loading Model: {self.modelName}")
        
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", 
                                                        self.modelName, "saved_model"
                                                        ))
        print(f"Model {self.modelName} loaded sucessifully...\n")
>>>>>>> 821456f3be6b555845362a2e0c20c25eb026742b
