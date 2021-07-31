from micro_sleep.dao import State, Estimator, Activity
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import joblib


class Estimators:
    eye: KernelDensity = None
    mouth: KernelDensity = None

    def __init__(self, eye: KernelDensity, mouth: KernelDensity):
        self.eye = eye
        self.mouth = mouth

    def predict_prob(self, eye: list, mouth: list) -> (list, list):
        """
        Uses the estimator and computes the probability of occurrence of each sample in X
        :param eye: array of eye scores
        :param mouth: array of mouth scores
        :return: This finds the probability for every value in X
        """
        pdf_eye = np.exp(self.eye.score_samples(np.array(eye).reshape(-1, 1)))
        pdf_mouth = np.exp(self.eye.score_samples(np.array(mouth).reshape(-1, 1)))

        return pdf_eye, pdf_mouth


class ActivityModel:
    def __init__(self):
        self.event_list = list()
        self.estimators = {Estimator.EYE: None, Estimator.MOUTH: None}

    def add_data(self, data: Activity):
        self.event_list.append(data)

    def transform(self, X, **transform_params):
        pass

    def fit(self, X, y=None, **fit_params):
        """
        Fits a kernel density estimator for the samples of observed values.
        The best bandwidth and the best parametric model is chosen via a search
        :param X: sample values to be used for estimation
        :param y: dummy
        :param fit_params: dictionary of param values
        :return: estimated kernel
        """
        #

        grid = GridSearchCV(estimator=KernelDensity(), param_grid={'bandwidth': np.linspace(0.1, 1.0, 30),
                                     'kernel': ['gaussian', 'tophat', 'epanechnikov',
                                                'exponential', 'linear','cosine']}, cv=10)  # 10-fold
        grid.fit(X.reshape(-1, 1))
        kde = grid.best_estimator_
        return kde

    def predict_prob(self, estimator, X):
        """
        Uses the estimator and computes the probability of occurrence of each sample in X
        :param estimator: the estimator to use
        :param X: array of samples for which we need to predict the probability
        :return: This finds the probability for every value in X
        """
        pdf = np.exp(estimator.score_samples(X.reshape(-1,1)))
        return pdf

    def build_model(self):
        df = pd.DataFrame(self.event_list)
        df = df[df.person_state == State.FOCUSED]
        eye_scores = np.array(df['eye_score'])
        mouth_scores = np.array(df['mouth_score'])

        self.estimators[Estimator.EYE] = self.fit(eye_scores)
        self.estimators[Estimator.MOUTH] = self.fit(mouth_scores)

    def predict(self, data: list):
        result = list()
        eye = list(map(lambda x: x.eye_score, data))
        mouth = list(map(lambda x: x.mouth_score, data))
        eye_prob = self.predict_prob(self.estimators[Estimator.EYE], np.array(eye))
        mouth_prob = self.predict_prob(self.estimators[Estimator.MOUTH], np.array(mouth))
        return eye_prob, mouth_prob

    def save_estimators(self, file_name):
        joblib.dump(Estimators(self.estimators[Estimator.EYE], self.estimators[Estimator.MOUTH]), file_name)

    def get_event_list(self):
        return self.event_list




