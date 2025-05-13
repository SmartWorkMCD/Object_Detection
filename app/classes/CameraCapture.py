import cv2
import os
import sys
import time
from typing import Optional, Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from comms import init_broker, connect_broker
from config import MQTT_CONFIG
from classes.DetectionInfo import DetectionInfo
from classes.DualDetector import DualDetector


class CameraCapture:
    def __init__(
        self,
        camera_id: int = 0,
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
        show_output: bool = True,
        capture_interval: Optional[int] = None,
        save_video: bool = False,
        resolution: Tuple[int, int] = (1080, 720),  # Default for C925e
        fps: float = 30.0,
        output_dir: str = "data",
        codec: str = "avc1",  # H.264 codec for Pi hardware acceleration
        use_yolo: bool = False,
        use_rfdetr: bool = False,
    ):
        """Initialize camera capture with configurable settings.

        Args:
            camera_id: Camera device ID
            flip_vertical: Whether to flip camera vertically
            flip_horizontal: Whether to flip camera horizontally
            show_output: Whether to display camera output
            capture_interval: Interval (in seconds) to automatically capture images, None to disable
            save_video: Whether to save video output
            resolution: Video resolution as (width, height)
            fps: Frames per second for video recording
            output_dir: Base directory for saving data
            codec: Video codec to use for recording
            use_yolo: Whether to use YOLO for detection
            use_rfdetr: Whether to use RF-DETR for detection
        """
        self.camera_id = camera_id
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.show_output = show_output
        self.capture_interval = capture_interval
        self.save_video = save_video
        self.resolution = resolution
        self.fps = fps
        self.output_dir = output_dir
        self.codec = codec
        self.use_yolo = use_yolo
        self.use_rfdetr = use_rfdetr

        # Create output directories if they don't exist
        self.frames_dir = os.path.join(output_dir, "frames")
        self.videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        # Camera and video writer objects
        self.cap = None
        self.video_writer = None
        self.last_capture_time = 0

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_report_interval = 10  # Report FPS every 10 seconds
        self.last_fps_report = self.start_time

    def setup(self) -> bool:
        """Set up camera and video writer. Returns True if successful."""
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False

        # Logitech C925e specific settings
        # Set preferred resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Set framerate
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual resolution (may differ from requested)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = actual_fps  # Update fps to actual value

        print(f"Camera initialized: {actual_width}x{actual_height} at {actual_fps} FPS")

        # Update resolution with actual values for video writer
        self.resolution = (int(actual_width), int(actual_height))

        # Set up video writer if requested
        if self.save_video:
            timestamp = int(time.time())
            video_path = os.path.join(self.videos_dir, f"{timestamp}.mp4")

            # Use hardware-accelerated codecs available on Raspberry Pi
            fourcc = cv2.VideoWriter_fourcc(*self.codec)

            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, self.resolution
            )

            if not self.video_writer.isOpened():
                print(f"Warning: Failed to create video writer with codec {self.codec}")
                # Try fallback to common codec
                self.video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    self.fps,
                    self.resolution,
                )
                if not self.video_writer.isOpened():
                    print(
                        "Error: Could not initialize video writer with fallback codec"
                    )
                    self.save_video = False
                else:
                    print(f"Video recording to: {video_path} (fallback codec)")
            else:
                print(f"Video recording to: {video_path}")

        self.last_capture_time = time.time()
        self.start_time = time.time()
        return True

    def process_frame(self, frame):
        """Apply processing to frame."""
        # Apply flipping if configured
        if self.flip_vertical:
            frame = cv2.flip(frame, 0)
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return frame

    def capture_image(self, frame):
        """Save current frame as image."""
        timestamp = int(time.time())
        image_path = os.path.join(self.frames_dir, f"{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved to: {image_path}")
        self.last_capture_time = time.time()

    def update_fps_stats(self):
        """Calculate and report FPS periodically."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_report

        # Report FPS periodically
        if elapsed >= self.fps_report_interval:
            fps = round(self.frame_count / elapsed)
            print(f"Performance: {fps} FPS")
            self.frame_count = 0
            self.last_fps_report = current_time

    def run(self):
        """Main processing loop."""
        if not self.setup():
            return

        print("Camera started. Press 'q' to quit, 's' to save a frame.")

        client = init_broker()
        connect_broker(client)
        print("Connected to MQTT broker")

        detector = DualDetector(
            yolo_weights_path=(
                "../../ultralytics/runs/detect/train/weights/best.pt"
                if self.use_yolo
                else None
            ),
            rfdetr_model_path="../../models/model_2_2.pth" if self.use_rfdetr else None,
        )

        if self.use_yolo or self.use_rfdetr:
            detectors_in_use = []
            if self.use_yolo:
                detectors_in_use.append("YOLO")
            if self.use_rfdetr:
                detectors_in_use.append("RF-DETR")
            print(f"Using {', '.join(detectors_in_use)} for detection")

        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    # Try to reconnect
                    time.sleep(1)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera_id)
                    if not self.cap.isOpened():
                        print("Failed to reconnect to camera")
                        break
                    continue

                # Update performance stats
                self.update_fps_stats()

                # Process the frame
                frame = self.process_frame(frame)

                # Send to queue
                if self.use_yolo or self.use_rfdetr:
                    info = DetectionInfo()
                    preds = detector.process_frame(frame)

                    if self.use_yolo:
                        info.add_yolo(
                            preds["yolo"]["boxes"],
                            preds["yolo"]["scores"],
                            preds["yolo"]["classes"],
                        )
                    if self.use_rfdetr:
                        info.add_rfdetr(
                            preds["rfdetr"]["boxes"],
                            preds["rfdetr"]["scores"],
                            preds["rfdetr"]["labels"],
                        )

                    info.timestamp = time.time()
                    json_str = info.to_json()
                    client.publish(MQTT_CONFIG.BROKER_TOPIC, json_str)

                # Save video frame if enabled
                if self.save_video and self.video_writer is not None:
                    self.video_writer.write(frame)

                # Check for auto-capture based on interval
                current_time = time.time()
                if (
                    self.capture_interval is not None
                    and current_time - self.last_capture_time >= self.capture_interval
                ):
                    self.capture_image(frame)

                # Display frame if enabled (may impact performance on Pi)
                if self.show_output:
                    # Resize for display if resolution is high
                    if self.resolution[0] > 800:
                        display_frame = cv2.resize(
                            frame,
                            (800, int(800 * self.resolution[1] / self.resolution[0])),
                        )
                    else:
                        display_frame = frame

                    # Draw detections on the frame
                    if self.use_yolo or self.use_rfdetr:
                        display_frame = detector.visualize(display_frame, info)

                    cv2.imshow("Camera Feed", display_frame)

                # Handle keyboard commands
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quitting...")
                    break
                elif key == ord("s"):
                    self.capture_image(frame)

                time.sleep(max(0, 1 / self.fps - (time.time() - current_time)))
        finally:
            # Clean up resources
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

        # Report final stats
        total_time = time.time() - self.start_time
        print(f"Session duration: {total_time:.1f} seconds")
        print("Camera resources released")
