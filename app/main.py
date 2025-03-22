import cv2
import time

########################################

# Flip camera
FLIP_CAMERA_VERTICALLY = False
FLIP_CAMERA_HORIZONTALLY = True

# Show camera
SHOW_CAMERA_OUTPUT = True

# Capture images
CAPTURE_IMAGES_EVERY_SECOND = False
CAPTURE_IMAGES_ON_COMMAND = False
CAPTURING_IMAGES = CAPTURE_IMAGES_EVERY_SECOND or CAPTURE_IMAGES_ON_COMMAND

# Save video
SAVE_VIDEO = True

########################################


def main() -> None:
    cap = cv2.VideoCapture(0)

    if CAPTURING_IMAGES:
        last_saved_time = time.time()

    if SAVE_VIDEO:
        out = cv2.VideoWriter(
            f"data/videos/{int(time.time())}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (640, 480),
        )

    while cap.isOpened():
        ret, frame = cap.read()

        if FLIP_CAMERA_VERTICALLY:
            frame = cv2.flip(frame, 0)

        if FLIP_CAMERA_HORIZONTALLY:
            frame = cv2.flip(frame, 1)

        if not ret:
            break

        if SHOW_CAMERA_OUTPUT:
            cv2.imshow("Object Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if CAPTURING_IMAGES:
            current_time = time.time()

            if CAPTURE_IMAGES_EVERY_SECOND:
                if current_time - last_saved_time >= 1:
                    cv2.imwrite(f"data/frames/{int(time.time())}.jpg", frame)
                    last_saved_time = current_time
            elif CAPTURE_IMAGES_ON_COMMAND:
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    cv2.imwrite(f"data/frames/{int(time.time())}.jpg", frame)

        if SAVE_VIDEO:
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
