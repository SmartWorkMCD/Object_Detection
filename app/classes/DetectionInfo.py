import json
import pickle
import time


class DetectionInfo:
    """
    Stores detections from both YOLO and RF-DETR with timestamp.
    Supports flat dict, JSON, and pickle serialization.
    """

    def __init__(self):
        self.timestamp = time.time()
        # YOLO results
        self.yolo_boxes = []
        self.yolo_scores = []
        self.yolo_classes = []

        # RF-DETR results
        self.rfdetr_boxes = []
        self.rfdetr_scores = []
        self.rfdetr_labels = []

    def add_yolo(self, boxes, scores, classes):
        self.yolo_boxes.extend(boxes.tolist())
        self.yolo_scores.extend(scores.tolist())
        self.yolo_classes.extend(classes)

    def add_rfdetr(self, boxes, scores, labels):
        self.rfdetr_boxes.extend(boxes.tolist())
        self.rfdetr_scores.extend(scores.tolist())
        self.rfdetr_labels.extend(labels)

    def to_flat_dict(self):
        data = {"timestamp": self.timestamp}

        # YOLO
        for i, (b, s, c) in enumerate(
            zip(self.yolo_boxes, self.yolo_scores, self.yolo_classes)
        ):
            data[f"yolo_{i}_x1"] = b[0]
            data[f"yolo_{i}_y1"] = b[1]
            data[f"yolo_{i}_x2"] = b[2]
            data[f"yolo_{i}_y2"] = b[3]
            data[f"yolo_{i}_score"] = s
            data[f"yolo_{i}_class"] = c

        # RF-DETR
        for i, (b, s, cl) in enumerate(
            zip(self.rfdetr_boxes, self.rfdetr_scores, self.rfdetr_labels)
        ):
            data[f"rfdetr_{i}_x1"] = b[0]
            data[f"rfdetr_{i}_y1"] = b[1]
            data[f"rfdetr_{i}_x2"] = b[2]
            data[f"rfdetr_{i}_y2"] = b[3]
            data[f"rfdetr_{i}_confidence"] = s
            data[f"rfdetr_{i}_class_id"] = cl
        return data

    def to_json(self):
        return json.dumps(self.to_flat_dict())

    @classmethod
    def from_json(cls, json_str):
        flat = json.loads(json_str)
        obj = cls()
        obj.timestamp = flat.get("timestamp", time.time())

        # YOLO
        idx = 0
        while f"yolo_{idx}_x1" in flat:
            b = [
                flat[f"yolo_{idx}_x1"],
                flat[f"yolo_{idx}_y1"],
                flat[f"yolo_{idx}_x2"],
                flat[f"yolo_{idx}_y2"],
            ]
            s = flat[f"yolo_{idx}_score"]
            c = flat[f"yolo_{idx}_class"]
            obj.add_yolo([b], [s], [c])
            idx += 1

        # RF-DETR
        idx = 0
        while f"rfdetr_{idx}_x1" in flat:
            b = [
                flat[f"rfdetr_{idx}_x1"],
                flat[f"rfdetr_{idx}_y1"],
                flat[f"rfdetr_{idx}_x2"],
                flat[f"rfdetr_{idx}_y2"],
            ]
            s = flat[f"rfdetr_{idx}_confidence"]
            cl = flat[f"rfdetr_{idx}_class_id"]
            obj.rfdetr.append(b)
            obj.rfdetr_confidence.append(s)
            obj.rfdetr_class_id.append(int(cl))
            idx += 1
        return obj

    def to_pickle(self):
        return pickle.dumps(self.to_flat_dict())

    @classmethod
    def from_pickle(cls, pickle_bytes):
        flat = pickle.loads(pickle_bytes)
        return cls.from_json(json.dumps(flat))

    def __str__(self):
        return str(self.to_flat_dict())
