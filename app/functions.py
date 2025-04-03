import albumentations as A
import argparse
import cv2
import json
import numpy as np
import os
from config import COLOR_VALUES, OBJECTS_CONFIG


def apply_augmentation(frames_dir):
    """
    Apply augmentations to frames and corresponding masks.

    Args:
        frames_dir (str): Path to the directory containing frame images
    """
    video_filename = os.path.basename(frames_dir)
    masks_dir = os.path.join("data", "masks")
    mask = cv2.imread(os.path.join(masks_dir, f"{video_filename}.png"))

    if mask is None:
        print(f"Error: Mask not found for {video_filename}.png")
        return

    # Get all image files, sorted by their numeric filename
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=frame_key,
    )

    # If no frames found
    if not frame_files:
        print(f"Error: No image files found in {frames_dir}")
        return

    # Create output directories if they don't exist
    augmented_frames_path = os.path.join(
        "data", "augmented_data", "frames", video_filename
    )
    augmented_masks_path = os.path.join(
        "data", "augmented_data", "masks", video_filename
    )
    os.makedirs(augmented_frames_path, exist_ok=True)
    os.makedirs(augmented_masks_path, exist_ok=True)

    # List to store multiple augmentation pipelines
    augmentation_pipeline = A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.3),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1.0, p=0.5),
                ],
                p=0.3,
            ),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
            ),
        ]
    )

    for frame_file in frame_files:
        # Get the full path of the frame file
        frame_path = os.path.join(frames_dir, frame_file)

        # Read the frame
        frame = cv2.imread(frame_path)

        # Apply augmentations multiple times to increase dataset size
        for i in range(3):  # Apply augmentations 3 times per frame
            augmented = augmentation_pipeline(image=frame, mask=mask)
            aug_frame = augmented["image"]
            aug_mask = augmented["mask"]

            # Generate a unique filename for each augmented version
            frame_name = os.path.splitext(frame_file)[0]
            aug_frame_filename = os.path.join(
                augmented_frames_path, f"frame_{frame_name}_aug_{i}.jpg"
            )
            aug_mask_filename = os.path.join(
                augmented_masks_path, f"frame_{frame_name}_aug_{i}.png"
            )

            # Save augmented images
            cv2.imwrite(aug_frame_filename, aug_frame)
            cv2.imwrite(aug_mask_filename, aug_mask)


# ! TODO: Update this function to use the new augmented masks
def create_annotations():
    """Generate annotations for YOLO and RF-DETR models."""
    # Create output directory if it doesn't exist
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # Create annotations file
    annotations_file = os.path.join(output_dir, "annotations.json")

    # Create annotations dictionary for YOLO and RF-DETR
    annotations = {"YOLO": [], "RF-DETR": []}

    # Process all object config files
    for video_filename, config in OBJECTS_CONFIG.items():
        # Get video resolution
        video_path = os.path.join("data", "videos", f"{video_filename}.mp4")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Process each object in the config
        for x1, y1, x2, y2, color in config:
            # Calculate object center and dimensions
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            obj_width = (x2 - x1) / width
            obj_height = (y2 - y1) / height

            # Add annotation to YOLO format
            annotations["YOLO"].append(
                {
                    "video": video_filename,
                    "color": color,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": obj_width,
                    "height": obj_height,
                }
            )

            # Add annotation to RF-DETR format
            annotations["RF-DETR"].append(
                {
                    "video": video_filename,
                    "color": color,
                    "bbox": [x1, y1, x2, y2],
                }
            )

    # Save annotations to file
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=4)


def create_mask(frames_dir):
    """
    Create a color mask for the objects in the video based on the config file.

    Args:
        frames_dir (str): Path to the directory containing frame images
    """
    # Retrieve the config for the video
    video_filename = os.path.basename(frames_dir)
    config = OBJECTS_CONFIG.get(video_filename, None)

    # Validate config
    if config is None:
        print(f"Error: No config found for video {video_filename}")
        return

    # Get all image files, sorted by their numeric filename
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=frame_key,
    )

    # If no frames found
    if not frame_files:
        print(f"Error: No image files found in {frames_dir}")
        return

    # Read the first frame to get the dimensions
    frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))

    # Create the mask
    mask = np.zeros(frame.shape, dtype=np.uint8)

    # Apply color masks
    for x1, y1, x2, y2, color in config:
        cv2.rectangle(mask, (x1, y1), (x2, y2), COLOR_VALUES[color][::-1], 1)

    # Create output directory if it doesn't exist
    output_dir = os.path.join("data", "masks")
    os.makedirs(output_dir, exist_ok=True)

    # Save the mask
    mask_filename = os.path.join(output_dir, f"{video_filename}.png")
    cv2.imwrite(mask_filename, mask)


def extract_video_frames(video_path):
    """
    Extract frames from a video file and save them in a numbered sequence.

    Args:
        video_path (str): Full path to the input video file
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

    # Release video capture object
    cap.release()

    # Print summary
    print(f"Extraction complete. Saved {extracted_count} frames from {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Video details: {width}x{height} at {fps:.2f} FPS")


def frame_key(filename):
    """Key function for sorting frame filenames by numeric order."""
    try:
        return int(os.path.splitext(filename)[0])
    except ValueError:
        return float("inf")


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


def parse_camera_arguments():
    """Parse command line arguments for the camera capture script."""
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


def parse_util_arguments():
    """Parse command line arguments for the utility script."""
    parser = argparse.ArgumentParser(description="Utility functions for video frames")
    parser.add_argument(
        "--apply-augmentation",
        action="store_true",
        help="Apply augmentations to the dataset",
    )
    parser.add_argument(
        "--create-annotations",
        action="store_true",
        help="Generate annotations for YOLO and RF-DETR models",
    )
    parser.add_argument(
        "--create-masks",
        action="store_true",
        help="Create color masks for each object in the config file",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from each video in the data/videos directory",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove duplicate frames from each extracted frames directory",
    )
    parser.add_argument(
        "--renumber-frames",
        action="store_true",
        help="Renumber frames sequentially in each extracted frames directory (after manual removal)",
    )

    return parser.parse_args()


def process_frame_directories(directory, process_function, *args):
    """Helper function to process all frame directories."""
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isdir(file_path):
            process_function(file_path, *args)


def process_videos_in_directory(directory, process_function, *args):
    """Helper function to process all videos in a directory."""
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            file_path = os.path.join(directory, file)
            process_function(file_path, *args)


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
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=frame_key,
    )

    # If no frames found
    if not frame_files:
        print(f"Error: No image files found in {frames_dir}")
        return

    # Track unique frames
    unique_frames = {}
    removed_frames = []

    # Process frames
    for frame_file in frame_files:
        # Get the full path of the frame file
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
    # Get all image files, sorted by their numeric filename
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=frame_key,
    )

    # If no frames found
    if not frame_files:
        print(f"Error: No image files found in {frames_dir}")
        return

    # Rename files sequentially
    for new_index, old_filename in enumerate(frame_files):
        old_path = os.path.join(frames_dir, old_filename)
        new_filename = f"{new_index}.jpg"
        new_path = os.path.join(frames_dir, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
