import json
import pickle
import time


class DetectionInfo:
    """
    Stores detections from both YOLO and RF-DETR, along with a timestamp.
    Provides serialization to dict, JSON, and pickle.
    """

    def __init__(self):
        self.timestamp = time.time()
        # Structure to hold per-frame results
        self.yolo_boxes = []  # List of [x1,y1,x2,y2]
        self.yolo_scores = []  # List of floats
        self.yolo_classes = []  # List of class names
        self.rf_boxes = []  # List of normalized [cx,cy,w,h]
        self.rf_scores = []  # List of floats
        self.rf_labels = []  # List of ints

    def add_yolo(self, boxes, scores, classes):
        self.yolo_boxes.extend(boxes.tolist())
        self.yolo_scores.extend(scores.tolist())
        self.yolo_classes.extend(classes)

    def add_rf_detr(self, boxes, scores, labels):
        self.rf_boxes.extend(boxes.tolist())
        self.rf_scores.extend(scores.tolist())
        self.rf_labels.extend(labels.tolist())

    def to_flat_dict(self):
        """Return all data in a flat dict with descriptive keys."""
        data = {"timestamp": self.timestamp}
        # YOLO entries
        for i, (b, s, c) in enumerate(
            zip(self.yolo_boxes, self.yolo_scores, self.yolo_classes)
        ):
            data[f"yolo_{i}_x1"] = b[0]
            data[f"yolo_{i}_y1"] = b[1]
            data[f"yolo_{i}_x2"] = b[2]
            data[f"yolo_{i}_y2"] = b[3]
            data[f"yolo_{i}_score"] = s
            data[f"yolo_{i}_class"] = c
        # RF-DETR entries
        for i, (b, s, l) in enumerate(
            zip(self.rf_boxes, self.rf_scores, self.rf_labels)
        ):
            data[f"rf_{i}_cx"] = b[0]
            data[f"rf_{i}_cy"] = b[1]
            data[f"rf_{i}_w"] = b[2]
            data[f"rf_{i}_h"] = b[3]
            data[f"rf_{i}_score"] = s
            data[f"rf_{i}_label"] = int(l)
        return data

    def to_json(self):
        """Serialize to JSON string."""
        return json.dumps(self.to_flat_dict())

    @classmethod
    def from_json(cls, json_str):
        """Deserialize from JSON string to DetectionInfo."""
        flat = json.loads(json_str)
        obj = cls()
        obj.timestamp = flat.get("timestamp", time.time())
        # Parse YOLO
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
        # Parse RF-DETR
        idx = 0
        while f"rf_{idx}_cx" in flat:
            b = [
                flat[f"rf_{idx}_cx"],
                flat[f"rf_{idx}_cy"],
                flat[f"rf_{idx}_w"],
                flat[f"rf_{idx}_h"],
            ]
            s = flat[f"rf_{idx}_score"]
            l = flat[f"rf_{idx}_label"]
            obj.add_rf_detr([b], [s], [l])
            idx += 1
        return obj

    def to_pickle(self):
        """Serialize to pickle bytes."""
        return pickle.dumps(self.to_flat_dict())

    @classmethod
    def from_pickle(cls, pickle_bytes):
        """Deserialize from pickle bytes to DetectionInfo."""
        flat = pickle.loads(pickle_bytes)
        return cls.from_json(json.dumps(flat))

    def __str__(self):
        return str(self.to_flat_dict())
