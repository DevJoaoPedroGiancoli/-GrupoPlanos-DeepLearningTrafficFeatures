from Detector import *


#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#classFile = r"real_time_object_detection_cpu\model_data\coco.names"
#videoPath = r"real_time_object_detection_cpu\model_data\ERS122_teste.wmv"


detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictVideo(videoPath)