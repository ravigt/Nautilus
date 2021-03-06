from micro_sleep.config_loader import load_config
import cv2 as cv
from micro_sleep.dao import State, Activity
from micro_sleep.ml import Estimators
import joblib

from micro_sleep.detector import ActivityDetector

if __name__ == '__main__':
    config = load_config("../config.ini")

    estimators: Estimators = joblib.load(config.estimator_model_path)
    activity_detector = ActivityDetector(config.face_model_path)
    video_stream = cv.VideoCapture(config.video_file)

    frame_count = 0

    while video_stream.isOpened():
        ret, orig_frame = video_stream.read()

        if orig_frame is None:
            break
        frame = estimators.scale_frame(orig_frame)

        face_location = activity_detector.face_localizer(frame_count, frame)
        activity: Activity = activity_detector.detector(frame_count, face_location, frame)

        if activity.person_state == State.FOCUSED:
            # when the driver is looking straight
            eye, mouth = estimators.predict_prob([activity.eye_score], [activity.mouth_score])
            image = cv.rectangle(frame, (face_location.rect.left(), face_location.rect.top()),
                                 (face_location.rect.right(), face_location.rect.bottom()),
                                 (255, 0, 0), 2)
            print(eye, mouth)

            if eye < 0.04:
                image = cv.putText(image, "Drowsy", (face_location.rect.right(), face_location.rect.bottom()), cv.FONT_HERSHEY_COMPLEX, 0.25, (0, 0,
                                                                                                                                            255))
                cv.imshow("local view", image)
                cv.waitKey()
            else:
                cv.imshow("local view", image)

                cv.waitKey(1)

        else:
            # when the driver is distracted or looking out of the window
            cv.imshow("local view", frame)
            cv.waitKey(1)

        # the activity.score needs to be processed as a time series sequence to accurately the probability that the driver is drowsy/sleepy
        print("Frame: {} Person State:{} Eye Score:{} Mouth Score: {} ".format(activity.frame_number, activity.person_state, activity.eye_score,
                                                                              activity.mouth_score))

