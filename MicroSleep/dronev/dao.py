from collections import namedtuple

DISTRACTION = 'D'
FOCUSED = 'F'

FaceLocation = namedtuple('FaceLocation', ['frame_number', 'top_left', 'bottom_right'])
Activity = namedtuple('Activity', ['frame_number', 'type', 'face_location'])
