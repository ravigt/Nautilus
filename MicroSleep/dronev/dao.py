from collections import namedtuple

DISTRACTION = 'D'
FOCUSED = 'F'

FaceLocation = namedtuple('FaceLocation', ['frame_number', 'rect'])
Activity = namedtuple('Activity', ['frame_number', 'type', 'face_location'])
