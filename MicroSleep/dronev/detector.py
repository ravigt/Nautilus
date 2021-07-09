from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import dlib
from dao import FaceLocation, Activity, FOCUSED, DISTRACTION


class Detector(ABC):

    @abstractmethod
    def face_localizer(self, frame_number, frame) -> FaceLocation:
        pass

    @abstractmethod
    def feature_extractor(self, frame_number, face_location, frame):
        pass

    @abstractmethod
    def classifier(self, frame_number, face_location: FaceLocation, frame) -> Activity:
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
        det = self.hog_face_detector(frame, 1)
        if len(det) != 1:
            return None
        return FaceLocation(frame_number=frame_number, rect=det[0])

    def classifier(self, frame_number, face_location, frame) -> Activity:
        activity_type = DISTRACTION if face_location is None else FOCUSED
        return Activity(frame_number=frame_number, type=activity_type, face_location=face_location)

    def feature_extractor(self, frame_number, face_location: FaceLocation, frame):
        shapes = self.landmark_extractor(frame, face_location.rect)
        return shapes
