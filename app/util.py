import sys
from functions import (
    create_annotations,
    create_mask,
    extract_video_frames,
    parse_util_arguments,
    process_frame_directories,
    process_videos_in_directory,
    remove_duplicate_frames,
    renumber_frames,
)


def main():
    if len(sys.argv) < 2:
        print("Error: Missing command line arguments.")
        return

    args = parse_util_arguments()

    if args.create_annotations:
        # Create annotations for the models
        try:
            create_annotations()
            print("Annotations created successfully.")
        except Exception as e:
            print(f"Error creating annotations: {e}")

    if args.create_masks:
        # Create color masks for all objects in the config file
        try:
            process_frame_directories("data/frames", create_mask)
            print("Color masks created successfully.")
        except Exception as e:
            print(f"Error creating color masks: {e}")

    if args.extract_frames:
        # Extract frames from all videos in the data/videos directory
        try:
            process_videos_in_directory("data/videos", extract_video_frames)
            print("Frames extracted successfully.")
        except Exception as e:
            print(f"Error extracting frames from videos in 'data/videos': {e}")

    if args.remove_duplicates:
        # Remove duplicate frames from all extracted frames directories
        try:
            process_frame_directories("data/frames", remove_duplicate_frames)
            print("Duplicate frames removed successfully.")
        except Exception as e:
            print(f"Error removing duplicate frames in 'data/frames': {e}")

    if args.renumber_frames:
        # Renumber frames in all extracted frames directories
        try:
            process_frame_directories("data/frames", renumber_frames)
            print("Frames renumbered successfully.")
        except Exception as e:
            print(f"Error renumbering frames in 'data/frames': {e}")


if __name__ == "__main__":
    main()
