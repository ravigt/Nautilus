from collections import namedtuple
from enum import Enum

DISTRACTION = 'D'
FOCUSED = 'F'
SLEEPY = 'S'

FaceLocation = namedtuple('FaceLocation', ['frame_number', 'rect'])
Activity = namedtuple('Activity', ['frame_number', 'person_state', 'eye_mouth_state', 'eye_score', 'mouth_score'])
EyeMouth = namedtuple('EyeMouth', ['left_eye', 'right_eye', 'mouth'])


class State(Enum):
    UNK = 1
    DETECTED = 2
    DISTRACTED = 3
    FOCUSED = 4
