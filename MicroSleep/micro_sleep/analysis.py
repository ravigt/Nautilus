from micro_sleep.config_loader import load_config
import cv2 as cv
from micro_sleep.dao import State, Activity
from micro_sleep.ml import ActivityModel
import os

from micro_sleep.detector import ActivityDetector
import joblib

activity_model: ActivityModel = joblib.load("activity_model.jl")

eye_score = list(map(lambda x: x.eye_score, activity_model.event_list))
mouth_score = list(map(lambda x: x.mouth_score, activity_model.event_list))

eye_prob, mouth_prob = activity_model.predict_prob(eye_score, mouth_score)
eye_dist = sorted(list(zip(eye_prob, eye_score)), key=lambda x: x[1])
mouth_dist = sorted(list(zip(mouth_prob, list(map(lambda x: x.mouth_score, activity_model.event_list)))), key=lambda x: x[1])
fig, ax = plt.subplots(2, 1)
ax[0].plot(list(map(lambda x: x[1], mouth_dist)), list(map(lambda x: x[0], mouth_dist)))
ax[0].set_title("Mouth size")
ax[1].plot(list(map(lambda x: x[1], eye_dist)), list(map(lambda x: x[0], eye_dist)))
ax[1].set_title("Eye size")
plt.show()