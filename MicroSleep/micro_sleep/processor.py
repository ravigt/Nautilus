from micro_sleep.config_loader import load_config
import cv2 as cv
from micro_sleep.dao import State, Activity
from micro_sleep.ml import ActivityModel
import os

from micro_sleep.detector import ActivityDetector
import joblib

if __name__ == '__main__':
    config = load_config("../config.ini")

    activity_model = ActivityModel()
    activity_list = list()
    activity_detector = ActivityDetector(config.face_model_path)

    for video in os.listdir(config.video_path):
        if video.endswith(".mp4"):
            video_file = os.path.join(config.video_path, video)
        else:
            continue
        video_stream = cv.VideoCapture(video_file)

        frame_count = 0

        while video_stream.isOpened():
            ret, frame = video_stream.read()

            if frame is None:
                break

            face_location = activity_detector.face_localizer(frame_count, frame)
            activity: Activity = activity_detector.detector(frame_count, face_location, frame)

            if activity.person_state == State.FOCUSED:
                # when the driver is looking straight
                image = cv.rectangle(frame, (face_location.rect.left(), face_location.rect.top()),
                                     (face_location.rect.right(), face_location.rect.bottom()),
                (255, 0, 0), 2)

            # the activity.score needs to be processed as a time series sequence to accurately the probability that the driver is drowsy/sleepy
            print("Frame: {} Person State:{} Eye Score:{} Mouth Score: {}".format(activity.frame_number, activity.person_state, activity.eye_score,
                                                                                  activity.mouth_score))
            activity_list.append(activity)
            frame_count += 1
            activity_model.add_data(activity)
    activity_model.build_model()
    # save the activity_model so that plotting ana analysis on data points can be performed
    joblib.dump(activity_model, config.activity_model_path)
    # save the estimators so that a detector can use the same
    activity_model.save_estimators(config.estimator_model_path)
    print(joblib.load(config.activity_model_path).get_event_list())
