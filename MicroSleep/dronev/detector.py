from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import dlib
from dao import FaceLocation, EyeMouth, Activity, FOCUSED, DISTRACTION, SLEEPY


class Detector(ABC):

    @abstractmethod
    def face_localizer(self, frame_number, frame) -> FaceLocation:
        pass

    @abstractmethod
    def eye_width_score(self, frame_number, eye_mouth: EyeMouth):
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
            eye_mouth = self.eye_mouth_coordinates(frame_number, face_location, frame)
            eye_width = self.eye_width_score(frame_number, eye_mouth)
            activity_type = FOCUSED
            score = eye_width
        return Activity(frame_number=frame_number, type=activity_type, score=score)

    def eye_mouth_coordinates(self, frame_number, face_location: FaceLocation, frame) -> EyeMouth:
        points = self.landmark_extractor(frame, face_location.rect)
        if points.num_parts != 68:
            return None
        left_eye_boundary = [points.part(37), points.part(38), points.part(39), points.part(40), points.part(41), points.part(42)]
        right_eye_boundary = [points.part(43), points.part(44), points.part(45), points.part(46), points.part(47), points.part(48)]
        mouth_boundary = [points.part(49), points.part(50), points.part(51), points.part(52), points.part(53), points.part(54), points.part(55),
                          points.part(56), points.part(57), points.part(58), points.part(59), points.part(60)]
        return EyeMouth(left_eye=left_eye_boundary, right_eye=right_eye_boundary, mouth=mouth_boundary)

    def eye_width_score(self, frame_number, eye_mouth: EyeMouth):
        """
        This is a heuristic over and above a trained eye-detector
        It computes the ratio of distance of the upper eyelash and the lower eyelash to the width of the eye. This is computed for each eye
        separately. The ratios for each eye is returned. If the driver is sleepy the ratio should be close to 0 else the ratio should be high.
        What is high and what is low is dependent on the face of the driver. Hence the ratios should be processed as a time-series of ratios and an
        anomaly detector should be built using the ratios.

        :param frame_number: the frame number that is being processed
        :param eye_mouth: the location of the face in the frame
        :return: score for the eye width
        """

        lec = eye_mouth.left_eye
        rec = eye_mouth.right_eye

        DL1 = np.sqrt((lec[1].x-lec[5].x)**2 + (lec[1].y - lec[5].y)**2)
        DL2 = np.sqrt((lec[2].x-lec[4].x)**2 + (lec[2].y - lec[4].y)**2)
        LEW = np.sqrt((lec[0].x-lec[3].x)**2 + (lec[0].y - lec[3].y)**2)

        DR1 = np.sqrt((rec[1].x-rec[5].x)**2 + (rec[1].y - rec[5].y)**2)
        DR2 = np.sqrt((rec[2].x-rec[4].x)**2 + (rec[2].y - rec[4].y)**2)
        REW = np.sqrt((rec[0].x-rec[3].x)**2 + (rec[0].y - rec[3].y)**2)

        R1 = (DL1+DL2)/2
        R2 = (DR1+DR2)/2

        mean_eye_width = (R1+R2)/2
        return mean_eye_width
