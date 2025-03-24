from functions import extract_video_frames, remove_duplicate_frames
import os


def main():
    # Extract frames from all videos in the data/videos directory
    for video in os.listdir("data/videos"):
        if video.endswith(".mp4"):
            video_path = os.path.join("data", "videos", video)
            extract_video_frames(video_path)

    # Remove duplicate frames from all extracted frames directories
    for file in os.listdir("data/frames"):
        if os.path.isdir(os.path.join("data", "frames", file)):
            frames_dir = os.path.join("data", "frames", file)
            remove_duplicate_frames(frames_dir)


if __name__ == "__main__":
    main()
