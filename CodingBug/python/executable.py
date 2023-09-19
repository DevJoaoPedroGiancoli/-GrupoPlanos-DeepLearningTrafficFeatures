from Model import Model
from Detector import Detector

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
classFile = r"CodingBug\coco.names"
imagePath = r"CodingBug\Images\traffic_images_1.jpg"

# Crie uma instância da classe Model
model_instance = Model()

# Crie uma instância da classe Detector, passando a instância da classe Model como argumento
detector_instance = Detector(model_instance)

# Carregando o modelo
model_instance.downloadModel(modelURL)
model_instance.loadModel()

# Agora você pode usar tanto model_instance quanto detector_instance conforme necessário
detector_instance.predictImage(imagePath)
