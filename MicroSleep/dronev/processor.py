from config_loader import load_config
import cv2 as cv
from dao import FOCUSED, DISTRACTION
import logging

from detector import ActivityDetector

if __name__ == '__main__':
    config = load_config("../config.ini")

    video_stream = cv.VideoCapture(config.video_path)
    activity_detector = ActivityDetector()

    frame_count = 0
    while video_stream.isOpened():
        ret, frame = video_stream.read()
        face_location = activity_detector.face_localizer(frame_count, frame)
        activity = activity_detector.classifier(frame_count, face_location, frame)

        if activity.type == FOCUSED:
            image = cv.rectangle(frame, activity.face_location.top_left, activity.face_location.bottom_right, (255, 0, 0), 2)
            cv.imshow("local view", image)
            cv.waitKey(1)
        else:
            cv.imshow("local view", frame)
            cv.waitKey(1)

        print("Frame: {} Activity:{}".format(activity.frame_number, activity.type))
        frame_count += 1

