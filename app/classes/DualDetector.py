import cv2
from DetectionInfo import DetectionInfo
import os
import torch
from ultralytics import YOLO


class DualDetector:
    def __init__(
        self,
        yolo_weights_path: str,
        rf_detr_model_path: str,
    ):
        # Initialize YOLO model
        self.yolo = YOLO(os.path.join(os.path.dirname(__file__), yolo_weights_path))
        self.yolo.to("cpu")

        # Initialize RF-DETR model
        self.rf_model = torch.load(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), rf_detr_model_path)
            ),
            map_location=torch.device("cpu"),
        )

    def process_frame(self, frame):
        """
        Runs both YOLO and RF-DETR on a frame.
        Returns a dict with 'yolo' and 'rf_detr' results.
        """
        # --- YOLO Inference ---
        yolo_results = self.yolo(frame, verbose=False)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()  # [N, 4]
        scores = yolo_results.boxes.conf.cpu().numpy()  # [N]
        classes = [
            yolo_results.names[int(c)] for c in yolo_results.boxes.cls.cpu().numpy()
        ]

        # --- RF-DETR Inference placeholder ---
        # info.add_rf_detr(boxes2, scores2, labels2)

        return {
            "yolo": {"boxes": boxes, "scores": scores, "classes": classes},
            # "rf_detr": {"boxes": boxes2, "scores": scores2, "labels": labels2},
        }

    def visualize(self, frame, info: DetectionInfo):
        """
        Draws stored detections onto image.
        Returns annotated image.
        """
        # Draw YOLO
        for _, (b, s, c) in enumerate(
            zip(info.yolo_boxes, info.yolo_scores, info.yolo_classes)
        ):
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

        # Draw RF-DETR
        h, w, _ = frame.shape
        for _, (b, s, l) in enumerate(
            zip(info.rf_boxes, info.rf_scores, info.rf_labels)
        ):
            cx, cy, bw, bh = b
            x1 = int((cx - 0.5 * bw) * w)
            y1 = int((cy - 0.5 * bh) * h)
            x2 = int((cx + 0.5 * bw) * w)
            y2 = int((cy + 0.5 * bh) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{l} {s:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        return frame
