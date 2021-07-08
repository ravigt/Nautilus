from config_loader import load_config
import cv2 as cv

from detector import DistractionDetector

if __name__ == '__main__':
    config = load_config("../config.ini")

    video_stream = cv.VideoCapture(config.video_path)
    distraction = DistractionDetector()

    while video_stream.isOpened():
        ret, frame = video_stream.read()
        cv.imshow("frame view", frame)
        cv.waitKey(1)
        result = distraction.face_localizer(frame)
        if result is not None:
            image = cv.rectangle(frame, result[0], result[1], (255, 0, 0), 2)
            cv.imshow("local view", frame)
            cv.waitKey(1)

