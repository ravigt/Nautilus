from dao import State, Activity
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


class ActivityModel(TransformerMixin):
    def __init__(self):
        super().__init__()

    event_list = list()

    def add_data(self, data: Activity):
        self.event_list.append(data)

    def transform(self, X, **transform_params):
        pass

    def fit(self, X, y=None, **fit_params):
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=10)  # 10-fold cross-validation
        grid.fit(X[:, None])
        kde = grid.best_estimator_
        return kde

    def plot_kde(self, estimator, X):
        pdf = np.exp(estimator.score_samples(X[:, None]))

        fig, ax = plt.subplots()
        ax.plot(X, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % estimator.bandwidth)
        # ax.hist(X, bins=10, fc='gray', histtype='stepfilled', alpha=0.3)
        ax.legend(loc='upper left')
        plt.pause(10)

    def show(self):
        df = pd.DataFrame(self.event_list)
        df = df[df.person_state == State.FOCUSED]
        eye_scores = df['eye_score']
        mouth_scores = df['mouth_score']

        eye_kde = self.fit(eye_scores)
        mouth_kde = self.fit(mouth_scores)

        self.plot_kde(eye_kde, eye_scores)

    def process(self):
        pass





