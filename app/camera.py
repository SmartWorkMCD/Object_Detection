from classes.CameraCapture import CameraCapture
from functions import parse_main_arguments


def main():
    """Main entry point."""
    try:
        # Check if running on Raspberry Pi
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read()
                if "Raspberry Pi" in model:
                    print(f"Detected: {model.strip('\0')}")
        except:
            pass

        # For backwards compatibility, use config from code if running directly
        if __name__ == "__main__":
            # Use command line arguments if provided, otherwise use defaults
            import sys

            if len(sys.argv) > 1:
                config = parse_main_arguments()
            else:
                config = {
                    "camera_id": 0,
                    "flip_vertical": False,
                    "flip_horizontal": False,
                    "show_output": True,
                    "capture_interval": None,  # CAPTURE_IMAGES_EVERY_SECOND
                    "save_video": False,
                    "resolution": (1080, 720),  # C925e default HD resolution
                    "fps": 30.0,
                    "output_dir": "data",
                    "codec": "avc1",
                    "send_to_queue": False,
                }

        camera = CameraCapture(**config)
        camera.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
