from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import dlib
from dao import FaceLocation, Activity, FOCUSED, DISTRACTION


class Detector(ABC):

    @abstractmethod
    def face_localizer(self, id, frame) -> FaceLocation:
        pass

    @abstractmethod
    def classifier(self, frame_number, face_location: FaceLocation, frame) -> Activity:
        pass


class ActivityDetector(Detector):

    frame_w, frame_h = 0, 0
    hog_face_detector = None

    def __init__(self):
        super().__init__()
        self.hog_face_detector = dlib.get_frontal_face_detector()

    def face_localizer(self, id, frame) -> FaceLocation:
        det = self.hog_face_detector(frame, 1)
        if len(det) != 1:
            return None
        return FaceLocation(frame_number=id, top_left=(det[0].left(), det[0].top()), bottom_right=(det[0].right(), det[0].bottom()))

    def classifier(self, frame_number, face_location, frame) -> Activity:
        activity_type = DISTRACTION if face_location is None else FOCUSED
        return Activity(frame_number=frame_number, type=activity_type, face_location=face_location)


