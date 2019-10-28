"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np


class DirectionalDataAnomalyDetection():
    def __init__(self):
        self.mean_val = None

    def fit(self, X):
        """Fit the model according to the given train data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
        """
        self.mean_val = X.mean(axis=0)

        return self

    def score(self, X):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """
        anomaly_score = 1 - X@self.mean_val
        return anomaly_score


if __name__ == '__main__':

    normal_data = np.loadtxt(
        "../input/normal_direction_data.csv", delimiter=",")
    error_data = np.loadtxt("../input/error_direction_data.csv", delimiter=",")

    model = DirectionalDataAnomalyDetection()
    model.fit(normal_data)
    pred = model.score(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
