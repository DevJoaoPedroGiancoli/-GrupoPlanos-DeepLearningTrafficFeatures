import os 
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file


class Model:
    def __init__(self) -> None:
        pass
    
    def downloadModel(self, modelURL):
        
        fileName = os.path.basename(modelURL)
        
        self.modelName = fileName[:fileName.index('.')]
        
        print(fileName)
        print(self.modelName)
        
        self.cacheDir = "./pretained_models"
        
        os.makedirs(self.cacheDir, exist_ok=True)
        os.makedirs(self.cacheDir, exist_ok=True)
        
        get_file(fname=fileName,
                 origin=modelURL,
                 cache_dir=self.cacheDir,
                 cache_subdir="checkpoints", extract=True)
        
    def loadModel(self):
        print("Loading Model" + self.modelName)
        tf.keras.backend.clear_session()
        
        self.model = tf.saved_model.load(os.path.join(self.cacheDir,
                                                        "checkpoints",
                                                        self.modelName,
                                                        "saved_model"))

        print(f"Model {self.modelName} loaded sucessfully...")
            