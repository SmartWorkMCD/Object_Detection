from functions import extract_video_frames
import os

def main():
    for video in os.listdir("data/videos"):
        if video.endswith(".mp4"):
            video_path = os.path.join("data", "videos", video)
            extract_video_frames(video_path)

if __name__ == "__main__":
    main()
