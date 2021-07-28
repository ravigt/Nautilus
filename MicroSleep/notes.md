# Outline of the approach

### Data capture conditions
1. faces of drivers vary based on position size and race
2. the camera position in the vehicle impacts the detection capability
3. most commonly low-light condition
4. stream of video frames

### Detection decisions
1. common approach is to use the eye size and mouth size indicators
2. size is a relative term as the structure of the eye and mouth vary based on camera positions and facial structure of the person's face

### Common Approaches
1. most common approach is to solve the problem as a supervised learning problem where many frame sequences of varying diversity are used to train a classifier
2. one needs examples of sequences where the person is alert, distracted (looking sideways) or drowsy. there are so many variations to these classes and even humans find it hard to apply a binary decision of these states.

### Approach taken 
* the problem is being solved for a specific case of driving in low-light conditions.
* the driver is either looking in front (at the road) or is looking sideways (outside the vehicle, turned back, looking down etc)
* first step is to perform detection of face to determine whether the person is looking front or not. This is a binary decision and can easily be performed using pre-trained HoG face model available as part of `dlib` python package. 

```
hog_face_detector = dlib.get_frontal_face_detector()
det = hog_face_detector(frame, 1)
if len(det) != 1:
	return 'Face not deteted -> implies driver is distracted in this frame'
```


* for all cases where we did detect a face detect the facial landmarks corresponding to the eyes and mouth. This is also easily performed using a pre-built model available in `dlib`

```
landmark_extractor = dlib.shape_predictor(shape_detector_model_path)
points = landmark_extractor(frame, face_location.rect)
```
* once the landmarks are obtained get the sizes of eye and mouth. Read the code to see how it is done.
* at this stage we get a time-series of the form `{frame:int, state:DISTRACTED|FOCUSED, mouth_size:float, eye_size:float}` 
* this time-series needs to be processed to detect the extent to which the driver is drowsy.
