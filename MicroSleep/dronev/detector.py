from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import dlib
from dao import FaceLocation, Activity, FOCUSED, DISTRACTION, SLEEPY


class Detector(ABC):

    @abstractmethod
    def face_localizer(self, frame_number, frame) -> FaceLocation:
        pass

    @abstractmethod
    def eye_state(self, frame_number, face_location, frame):
        pass

    @abstractmethod
    def detector(self, frame_number, face_location: FaceLocation, frame) -> Activity:
        pass


class ActivityDetector(Detector):

    frame_w, frame_h = 0, 0
    hog_face_detector = None
    landmark_extractor = None

    def __init__(self, shape_detector_model_path):
        super().__init__()
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.landmark_extractor = dlib.shape_predictor(shape_detector_model_path)

    def face_localizer(self, frame_number, frame) -> FaceLocation:
        try:
            det = self.hog_face_detector(frame, 1)
            if len(det) != 1:
                return None
            return FaceLocation(frame_number=frame_number, rect=det[0])
        except Exception as e:
            print(e)
            return None

    def detector(self, frame_number, face_location, frame) -> Activity:
        activity_type = FOCUSED
        score = 1
        if face_location is None:
            activity_type = DISTRACTION
            score = 1
        else:
            r1, r2 = self.eye_state(frame_number, face_location, frame)
            activity_type = FOCUSED
            score = (r1+r2)/2
        return Activity(frame_number=frame_number, type=activity_type, score=score)

    def eye_state(self, frame_number, face_location: FaceLocation, frame):
        """
        This is a heuristic over and above a trained eye-detector
        It computes the ratio of distance of the upper eyelash and the lower eyelash to the width of the eye. This is computed for each eye
        separately. The ratios for each eye is returned. If the driver is sleepy the ratio should be close to 0 else the ratio should be high.
        What is high and what is low is dependent on the face of the driver. Hence the ratios should be processed as a time-series of ratios and an
        anomaly detector should be built using the ratios.

        :param frame_number: the frame number that is being processed
        :param face_location: the location of the face in the frame
        :param frame: the grey scale version of the frame
        :return: tuple of scores for the left eye and right eye
        """
        points = self.landmark_extractor(frame, face_location.rect)
        if points.num_parts != 68:
            return None

        le_l = points.part(37)
        le_t1 = points.part(38)
        le_t2 = points.part(39)
        le_r = points.part(40)
        le_b2 = points.part(41)
        le_b1 = points.part(42)

        re_l = points.part(43)
        re_t1 = points.part(44)
        re_t2 = points.part(45)
        re_r = points.part(46)
        re_b2 = points.part(47)
        re_b1 = points.part(48)

        DL1 = np.sqrt((le_t1.x-le_b1.x)**2 + (le_t1.y - le_t2.y)**2)
        DL2 = np.sqrt((le_t2.x-le_b2.x)**2 + (le_t2.y - le_t2.y)**2)
        LEW = np.sqrt((le_l.x-le_r.x)**2 + (le_l.y - le_r.y)**2)

        DR1 = np.sqrt((re_t1.x-re_b1.x)**2 + (re_t1.y - re_t2.y)**2)
        DR2 = np.sqrt((re_t2.x-re_b2.x)**2 + (re_t2.y - re_t2.y)**2)
        REW = np.sqrt((re_l.x-re_r.x)**2 + (re_l.y - re_r.y)**2)

        R1 = (DL1+DL2)/(2*LEW)
        R2 = (DR1+DR2)/(2*REW)

        return R1, R2
