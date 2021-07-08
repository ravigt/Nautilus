from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import dlib


class Detector(ABC):

    @abstractmethod
    def face_localizer(self, frame):
        pass

    @abstractmethod
    def classifier(self, region):
        pass


class DistractionDetector(Detector):

    frame_w, frame_h = 0, 0
    hog_face_detector = None

    def __init__(self):
        super().__init__()
        self.hog_face_detector = dlib.get_frontal_face_detector()

    def face_localizer(self, frame):
        det = self.hog_face_detector(frame, 1)
        if len(det) != 1:
            return None
        top_left = (det[0].left(), det[0].top())
        bottom_right = (det[0].right(), det[0].bottom())
        return top_left, bottom_right

    def classifier(self, region):
        pass


