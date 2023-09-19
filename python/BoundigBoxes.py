import cv2
import tensorflow as tf
import numpy as np

class BoundingBoxes:
    def __init__(self) -> None:
        pass
    
    def createBoundigBox(self, image):
        
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype = tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]
        
        detections = self.model(inputTensor)
        
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()
        
        imH, imW, imC = image.shape
        
        bboxIdx = tf.image.non_max_supression(bboxs, classScores, max_output_size = 50, iout_threshold = 0.5)
        
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]
                
                classLabelText = self.classesLIST[classIndex]
                classColor = self.colorList[classIndex]
                
                displayText = '{}: {}%'.format(classLabelText, classConfidence)
                
                ymin, xmin, ymax, xmax = bbox
                
                print(ymin, xmin, ymax, xmax)
                
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                
                
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = classColor, thickness = 1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, classColor, 2)
                
                lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
                
                cv2.line(image, (xmin,ymin), (xmin + lineWidth, ymin), classColor, thickness = 5)

                
        return image