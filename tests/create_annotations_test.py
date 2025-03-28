import unittest
from unittest.mock import patch, MagicMock
import os
import json
from app.functions import create_annotations


# ! DOES NOT WORK


class TestCreateAnnotations(unittest.TestCase):
    @patch("app.functions.os.makedirs")
    @patch("app.functions.json.dump")
    @patch("app.functions.cv2.VideoCapture")
    @patch(
        "app.functions.OBJECTS_CONFIG",
        {"video1": [(10, 20, 30, 40, "red"), (50, 60, 70, 80, "blue")]},
    )
    def test_create_annotations(
        self, mock_video_capture, mock_json_dump, mock_makedirs
    ):
        # Mock video capture behavior
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda x: (
            1920 if x == 3 else 1080
        )  # Width=1920, Height=1080
        mock_video_capture.return_value = mock_cap

        # Call the function
        create_annotations()

        # Check if output directory was created
        mock_makedirs.assert_called_once_with("models", exist_ok=True)

        # Check if annotations were saved to file
        expected_annotations = {
            "YOLO": [
                {
                    "video": "video1",
                    "color": "red",
                    "center_x": 0.010416666666666666,
                    "center_y": 0.027777777777777776,
                    "width": 0.010416666666666666,
                    "height": 0.018518518518518517,
                },
                {
                    "video": "video1",
                    "color": "blue",
                    "center_x": 0.03125,
                    "center_y": 0.06481481481481481,
                    "width": 0.010416666666666666,
                    "height": 0.018518518518518517,
                },
            ],
            "RF-DETR": [
                {"video": "video1", "color": "red", "bbox": [10, 20, 30, 40]},
                {"video": "video1", "color": "blue", "bbox": [50, 60, 70, 80]},
            ],
        }
        mock_json_dump.assert_called_once_with(
            expected_annotations, unittest.mock.ANY, indent=4
        )


if __name__ == "__main__":
    unittest.main()
