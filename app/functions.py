import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Camera capture for Logitech C925e on Raspberry Pi"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--flip-v", action="store_true", help="Flip camera vertically")
    parser.add_argument(
        "--flip-h", action="store_true", help="Flip camera horizontally"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Don't show camera output"
    )
    parser.add_argument(
        "--interval", type=int, help="Interval in seconds to auto-capture images"
    )
    parser.add_argument("--save-video", action="store_true", help="Save video output")
    parser.add_argument(
        "--resolution", default="1080x720", help="Resolution in format WIDTHxHEIGHT"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Frames per second for video"
    )
    parser.add_argument(
        "--output-dir", default="data", help="Base directory for saving data"
    )
    parser.add_argument(
        "--codec",
        default="avc1",
        help="Video codec (avc1 for H.264 hardware acceleration)",
    )

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))

    return {
        "camera_id": args.camera,
        "flip_vertical": args.flip_v,
        "flip_horizontal": args.flip_h,
        "show_output": not args.no_display,
        "capture_interval": args.interval,
        "save_video": args.save_video,
        "resolution": (width, height),
        "fps": args.fps,
        "output_dir": args.output_dir,
        "codec": args.codec,
    }
