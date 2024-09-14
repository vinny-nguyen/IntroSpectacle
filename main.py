import numpy as np
import cv2 as cv
import time

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    tap = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.imshow('Live Feed', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            tap = 1
        elif key == 27:  # 'Esc' key to exit
            break

        if tap == 1:
            # Capture and save a picture
            image_filename = f'person.png'
            cv.imwrite(image_filename, frame)
            print(f"Image saved as {image_filename}")
            tap = 0
            continue

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()