import numpy as np

class Reader:
    def __init__(self) -> None:
        pass
    
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesLIST = f.read().splitlines()
            # Colors list
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesLIST), 3))
            
            print(len(self.classesLIST), len(self.colorList))