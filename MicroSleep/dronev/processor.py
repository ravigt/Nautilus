from config_loader import load_config
import cv2 as cv
from dao import FOCUSED, DISTRACTION, SLEEPY

from detector import ActivityDetector

if __name__ == '__main__':
    config = load_config("../config.ini")

    video_stream = cv.VideoCapture(config.video_path)
    activity_detector = ActivityDetector(config.model_path)

    frame_count = 0
    while video_stream.isOpened():
        ret, frame = video_stream.read()

        if frame is None:
            break

        face_location = activity_detector.face_localizer(frame_count, frame)
        activity = activity_detector.detector(frame_count, face_location, frame)

        if activity.type == FOCUSED or activity.type == SLEEPY:
            # when the driver is looking straight
            image = cv.rectangle(frame, (face_location.rect.left(), face_location.rect.top()),
                                 (face_location.rect.right(), face_location.rect.bottom()),
            (255, 0, 0), 2)
            cv.imshow("local view", image)
            cv.waitKey(1)
        else:
            # when the driver is distracted or looking out of the window
            cv.imshow("local view", frame)
            cv.waitKey(1)

        # the activity.score needs to be processed as a time series sequence to accurately the probability that the driver is drowsy/sleepy
        print("Frame: {} Activity:{} Score:{}".format(activity.frame_number, activity.type, activity.score))
        frame_count += 1

