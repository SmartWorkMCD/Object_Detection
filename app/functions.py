import argparse
import cv2
import numpy as np
import os


def extract_video_frames(video_path, output_dir=None, log=False):
    """
    Extract frames from a video file and save them in a numbered sequence.

    Args:
        video_path (str): Full path to the input video file
        output_dir (str, optional): Directory to save extracted frames.
                                    If None, uses a folder in the frames directory
                                    with the same name as the video file.
    """
    # Validate input video file
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine output directory
    if output_dir is None:
        # Create directory in frames folder with video filename (without extension)
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join("data", "frames", video_filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save frames
    extracted_count = 0
    while True:
        ret, frame = cap.read()

        # Break if no more frames
        if not ret:
            break

        # Generate frame filename with 6-digit zero-padded number
        frame_filename = os.path.join(output_dir, f"{extracted_count}.jpg")

        # Save frame
        cv2.imwrite(frame_filename, frame)
        extracted_count += 1

        # Optional: Print progress periodically
        if log and extracted_count % 100 == 0:
            print(f"Extracted {extracted_count}/{total_frames} frames")

    # Release video capture object
    cap.release()

    # Print summary
    print(f"Extraction complete. Saved {extracted_count} frames from {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Video details: {width}x{height} at {fps:.2f} FPS")


def hash_frame(frame, hash_size=8):
    """
    Create a perceptual hash of the frame to identify similar images.

    Args:
        frame (numpy.ndarray): Input image frame
        hash_size (int): Size of the hash (smaller = more lenient matching)

    Returns:
        str: Perceptual hash of the frame
    """
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # Compute the DCT (Discrete Cosine Transform)
    dct = cv2.dct(np.float32(resized))

    # Use the top-left corner of the DCT coefficients
    dct_low_freq = dct[:hash_size, :hash_size]

    # Compute the median
    median = np.median(dct_low_freq)

    # Create hash based on whether each coefficient is above/below median
    hash_bits = (dct_low_freq > median).flatten()

    # Convert to hex string for compact representation
    hash_hex = "".join(["1" if bit else "0" for bit in hash_bits])
    return hash_hex


def parse_arguments() -> dict:
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


def remove_duplicate_frames(frames_dir):
    """
    Remove duplicate frames from a directory, keeping only unique images.

    Args:
        frames_dir (str): Path to the directory containing frame images
    """
    # Validate input directory
    if not os.path.exists(frames_dir):
        print(f"Error: Directory {frames_dir} does not exist.")
        return

    # Get all image files, sorted by their numeric filename
    def frame_key(filename):
        try:
            return int(os.path.splitext(filename)[0])
        except ValueError:
            return float("inf")

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=frame_key,
    )

    # If no frames found
    if not frame_files:
        print(f"No image files found in {frames_dir}")
        return

    # Track unique frames
    unique_frames = {}
    removed_frames = []

    # Process frames
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)

        # Read the frame
        frame = cv2.imread(frame_path)

        # Skip if image can't be read
        if frame is None:
            print(f"Warning: Could not read {frame_file}")
            continue

        # Create a hash of the frame to identify duplicates
        frame_hash = hash_frame(frame)

        # If this frame is unique, keep it
        if frame_hash not in unique_frames:
            unique_frames[frame_hash] = frame_file
        else:
            # If duplicate, mark for removal
            removed_frames.append(frame_file)
            os.remove(frame_path)

    # Renumber remaining frames sequentially
    renumber_frames(frames_dir)

    # Print summary
    print(f"\nDuplicate Frame Removal Summary (video {os.path.basename(frames_dir)}):")
    print(f"Total original frames: {len(frame_files)}")
    print(f"Unique frames kept: {len(unique_frames)}")
    print(f"Frames removed: {len(removed_frames)}")


def renumber_frames(frames_dir):
    """
    Renumber frames sequentially starting from 0.

    Args:
        frames_dir (str): Path to the directory containing frame images
    """

    # Get all image files, sorted by their current filename
    def frame_key(filename):
        try:
            return int(os.path.splitext(filename)[0])
        except ValueError:
            return float("inf")

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=frame_key,
    )

    # Rename files sequentially
    for new_index, old_filename in enumerate(frame_files):
        old_path = os.path.join(frames_dir, old_filename)
        new_filename = f"{new_index}.jpg"
        new_path = os.path.join(frames_dir, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
