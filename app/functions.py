import albumentations as A
import argparse
import cv2
import numpy as np
import os
from app.config import COLOR_CLASSES, COLOR_VALUES, OBJECTS_CONFIG


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
        key=file_key,
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
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussNoise(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomShadow(p=0.3),
            A.ISONoise(p=0.2),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                },
                rotate=(-5, 5),
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


def create_annotations():
    """Generate annotations in class_id center_x center_y width height format."""
    # Create output directories if they don't exist
    os.makedirs("data/annotations", exist_ok=True)

    # Get augmented masks directory
    augmented_masks_dir = os.path.join("data", "augmented_data", "masks")

    # Validate augmented masks directory
    if not os.path.exists(augmented_masks_dir):
        print(f"Error: Augmented masks directory {augmented_masks_dir} does not exist.")
        return

    for masks_dir in os.listdir(augmented_masks_dir):
        masks_dir_path = os.path.join(augmented_masks_dir, masks_dir)

        # Get all mask files, sorted by their numeric filename
        mask_files = sorted(
            [f for f in os.listdir(masks_dir_path) if f.lower().endswith(".png")],
            key=file_key,
        )

        # If no masks found
        if not mask_files:
            print(f"Error: No mask files found in {masks_dir_path}")
            return

        # Iterate through each mask file
        for mask_file in mask_files:
            # Get the full path of the mask file
            mask_path = os.path.join(masks_dir_path, mask_file)

            # Read the mask
            mask = cv2.imread(mask_path)

            # Check if the mask is valid
            if mask is None:
                print(f"Error: Mask {mask_file} could not be read.")
                continue

            # Convert the mask to RGB format
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # Apply dilation since lines are thin
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Get the dimensions of the mask
            height, width, _ = mask.shape

            # Initialize an empty list to store annotations
            annotations = []

            # Iterate through each color in the mask
            for color_name, color_value in COLOR_VALUES.items():
                # Get the class ID for the color
                class_id = COLOR_CLASSES[color_name]

                # Define lower and upper bounds for color matching with tolerance
                lower = np.array([max(c - 10, 0) for c in color_value])
                upper = np.array([min(c + 10, 255) for c in color_value])
                binary_mask = cv2.inRange(mask, lower, upper)

                # Find contours in the binary mask
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Filter contours to ensure only valid objects are considered
                valid_contours = [
                    contour for contour in contours if cv2.contourArea(contour) > 10
                ]

                # Check if we have at least 2 contours; if more, choose the two largest.
                if len(valid_contours) < 2:
                    continue
                elif len(valid_contours) > 2:
                    valid_contours = sorted(
                        valid_contours, key=cv2.contourArea, reverse=True
                    )[:2]

                # Iterate through each valid contour
                for contour in valid_contours:
                    # Calculate the bounding rectangle for the contour
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center coordinates and normalized values
                    center_x = (x + w / 2) / width
                    center_y = (y + h / 2) / height
                    norm_width = w / width
                    norm_height = h / height

                    # Append the annotation to the list
                    annotations.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )

            # Generate the annotation filename based on the mask filename
            annotation_filename = os.path.splitext(mask_file)[0] + ".txt"
            annotation_path = os.path.join(
                "data", "annotations", masks_dir, annotation_filename
            )

            # Ensure the directory for the annotation file exists
            os.makedirs(os.path.dirname(annotation_path), exist_ok=True)

            # Write the annotations to the file
            with open(annotation_path, "w") as f:
                f.write("\n".join(annotations))


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
        key=file_key,
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


def file_key(filename):
    """Key function for sorting by numeric order."""
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


def parse_util_arguments():
    """Parse command line arguments for the utility script."""
    parser = argparse.ArgumentParser(description="Utility functions for video frames")
    parser.add_argument(
        "--apply-augmentation",
        action="store_true",
        help="Apply augmentations to the dataset (requires frames and masks)",
    )
    parser.add_argument(
        "--create-annotations",
        action="store_true",
        help="Generate annotations for YOLO and RF-DETR models (requires masks)",
    )
    parser.add_argument(
        "--create-masks",
        action="store_true",
        help="Create color masks for each object in the config file (requires frames)",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from each video in the data/videos directory (requires videos)",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove duplicate frames from each extracted frames directory (requires frames)",
    )
    parser.add_argument(
        "--renumber-frames",
        action="store_true",
        help="Renumber frames sequentially in each extracted frames directory (after manual removal of frames)",
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
        key=file_key,
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
        key=file_key,
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
