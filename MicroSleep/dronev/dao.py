from collections import namedtuple

DISTRACTION = 'D'
FOCUSED = 'F'
SLEEPY = 'S'

FaceLocation = namedtuple('FaceLocation', ['frame_number', 'rect'])
Activity = namedtuple('Activity', ['frame_number', 'type', 'score'])
