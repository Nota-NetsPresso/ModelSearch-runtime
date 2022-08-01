from .base import Basemodel
import cv2
import numpy as np
class NPModel(Basemodel):
    def __init__(self):
        super().__init__()
        self.inputs = self.model_information().get('inputs')
        self.outputs = self.model_information().get('outputs')
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
        input_shape = self.inputs[0]['shape']
        dims = 'nchw' if input_shape[1] == 3 else 'nhwc'
        input_size = input_shape[2:4] if dims=='nchw' else input_shape[1:3]
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
        if self.inputs[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = self.inputs[0]['quantization']
            preprocessed_image = preprocessed_image / input_scale + input_zero_point
        preprocessed_image = preprocessed_image.astype(self.inputs[0]['dtype'])
        if dims == 'nchw':
            preprocessed_image = preprocessed_image.transpose(0,3,1,2)
        return [preprocessed_image]
    def postprocess(self, inference_results, **kwargs):
        conf_thres = kwargs.get('conf_thres', 0.25)
        iou_thres = kwargs.get('iou_thres', 0.60)
        result = inference_results[0].reshape((-1, len(self.classes)+5))
        result = np.expand_dims(result, axis=0)
        result = self.nms(result, conf_thres, iou_thres)
        result = self.normalize(result)
        self.print_result(result)
        return inference_results
    def compute_iou(self, box, boxes, box_area, boxes_area):
        assert boxes.shape[0] == boxes_area.shape[0]
        ys1 = np.maximum(box[0], boxes[:, 0])
        xs1 = np.maximum(box[1], boxes[:, 1])
        ys2 = np.minimum(box[2], boxes[:, 2])
        xs2 = np.minimum(box[3], boxes[:, 3])
        intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
        unions = box_area + boxes_area - intersections
        ious = intersections / unions
        return ious
    def non_max_suppression(self, boxes, scores, iou_thres):
        assert boxes.shape[0] == scores.shape[0]
        ys1 = boxes[:, 0]
        xs1 = boxes[:, 1]
        ys2 = boxes[:, 2]
        xs2 = boxes[:, 3]
        areas = (ys2 - ys1) * (xs2 - xs1)
        scores_indexes = scores.argsort().tolist()
        boxes_keep_index = []
        while len(scores_indexes):
            index = scores_indexes.pop()
            boxes_keep_index.append(index)
            if not len(scores_indexes):
                break
            ious = self.compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
            filtered_indexes = np.where(ious > iou_thres)[0]
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)
    def nms(self, prediction, conf_thres, iou_thres):
        prediction = prediction[prediction[..., 4] > conf_thres]
        boxes = self.xywh2xyxy(prediction[:, :4])
        res = self.non_max_suppression(boxes, prediction[:, 4], iou_thres)
        result_boxes = []
        for r in res:
            tmp = np.zeros(6)
            j = prediction[r, 5:].argmax()
            tmp[0] = boxes[r][0].item()
            tmp[1] = boxes[r][1].item()
            tmp[2] = boxes[r][2].item()
            tmp[3] = boxes[r][3].item()
            tmp[4] = prediction[r][4].item()
            tmp[5] = j
            result_boxes.append(tmp)
        return result_boxes
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    def normalize(self, boxes):
        input_shape = self.inputs[0]['shape']
        if not boxes:
            return boxes
        np_boxes = np.array(boxes)
        if np.all(np_boxes[:,:4] <= 1.0):
            return boxes
        for box in boxes:
            box[0] /= input_shape[1]
            box[1] /= input_shape[0]
            box[2] /= input_shape[1]
            box[3] /= input_shape[0]
        return boxes
    def print_result(self, result_label):
        print('--------------------------------------------------------------')
        for i, label in enumerate(result_label):
            detected = str(self.classes[int(label[5])])
            print('Detect ' + str(i+1) + '(' + str(detected) + ')')
            print('Confidence : ', label[4])
    def run(self, input_data, conf_thres = 0.25, iou_thres = 0.60, active_input_indexes = [0], interested_output_indexes = [0], num_threads = 1):
        return super().run(input_data, 
            conf_thres = conf_thres,
            iou_thres = iou_thres,
            active_input_indexes=active_input_indexes, 
            interested_output_indexes=interested_output_indexes, 
            num_threads=num_threads
        )
