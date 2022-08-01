from .base import Basemodel
import cv2
import numpy as np
class NPModel(Basemodel):
    def __init__(self):
        super().__init__()        
        self.inputs = self.model_information().get('inputs')
        self.outputs = self.model_information().get('outputs')
        self.input_size = self.model_information().get('input_size')
        self.classes = [
            "aeroplane", 
            "bicycle", 
            "bird", 
            "boat", 
            "bottle", 
            "bus", 
            "car", 
            "cat", 
            "chair", 
            "cow", 
            "diningtable",
            "dog", 
            "horse", 
            "motorbike", 
            "person", 
            "pottedplant", 
            "sheep", 
            "sofa", 
            "train", 
            "tvmonitor"
            ]
    def preprocess(self, image):
        input_size = self.input_size
        origin_h, origin_w, origin_c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r_w = input_size[1] / origin_w
        r_h = input_size[0] / origin_h
        if r_h > r_w:
            tw = input_size[1]
            th = int(r_w *  origin_h)
            tx1 = tx2 = 0
            ty1 = int((input_size[0] - th) / 2)
            ty2 = input_size[0] - th - ty1
        else:
            tw = int(r_h * origin_w)
            th = input_size[0]
            tx1 = int((input_size[1] - tw) / 2)
            tx2 = input_size[1] - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        preprocessed_image = np.ascontiguousarray(image)
        preprocessed_image = preprocessed_image.transpose(0,3,1,2)
        return [preprocessed_image]
    def postprocess(self, inference_results):
        result = inference_results.reshape((-1, len(self.classes)+5))
        return result
    def run(self, input_data):
        return super().run(input_data)