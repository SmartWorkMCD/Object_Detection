import cv2
from DetectionInfo import DetectionInfo
import os
from PIL import Image
import torch
from ultralytics import YOLO


class DualDetector:
    def __init__(
        self,
        yolo_weights_path: str,
        rfdetr_model_path: str,
    ):
        self.use_yolo = yolo_weights_path is not None
        self.use_rfdetr = rfdetr_model_path is not None

        if self.use_yolo:
            # Initialize YOLO model
            self.yolo = YOLO(os.path.join(os.path.dirname(__file__), yolo_weights_path))
            self.yolo.to("cpu")

        if self.use_rfdetr:
            # Initialize RF-DETR model
            self.rfdetr_model = torch.load(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), rfdetr_model_path)
                ),
                map_location=torch.device("cpu"),
            )
            self.rfdetr_model.model.device = torch.device("cpu")

    def process_frame(self, frame):
        """
        Runs both YOLO and RF-DETR on a frame.
        Returns a dict with 'yolo' and 'rfdetr' results.
        """
        if self.use_yolo:
            # --- YOLO Inference ---
            yolo_results = self.yolo(frame, conf=0.5, verbose=False)[0]
            boxes = yolo_results.boxes.xyxy.cpu().numpy()
            scores = yolo_results.boxes.conf.cpu().numpy()
            classes = [
                yolo_results.names[int(c)] for c in yolo_results.boxes.cls.cpu().numpy()
            ]

        if self.use_rfdetr:
            # --- RF-DETR Inference placeholder ---
            rfdetr_results = self.rfdetr_model.predict(
                Image.fromarray(frame), threshold=0.5
            )
            boxes2 = rfdetr_results.xyxy
            scores2 = rfdetr_results.confidence
            labels2 = [
                ["blue", "green", "orange", "red", "yellow"][i]
                for i in rfdetr_results.class_id
            ]

        results = {}
        if self.use_yolo:
            results["yolo"] = {"boxes": boxes, "scores": scores, "classes": classes}
        if self.use_rfdetr:
            results["rfdetr"] = {"boxes": boxes2, "scores": scores2, "labels": labels2}

        return results

    def visualize(self, frame, info: DetectionInfo):
        if self.use_yolo:
            # Draw YOLO
            for b, s, c in zip(info.yolo_boxes, info.yolo_scores, info.yolo_classes):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{c} {s:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        if self.use_rfdetr:
            # Draw RF-DETR
            for b, s, cl in zip(
                info.rfdetr_boxes, info.rfdetr_scores, info.rfdetr_labels
            ):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{cl} {s:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        return frame
