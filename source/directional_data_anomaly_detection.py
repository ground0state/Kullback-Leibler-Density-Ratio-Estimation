"""
Copyright © 2019 ground0state. All rights reserved.
"""
import numpy as np


class DirectionalDataAnomalyDetection():
    def __init__(self):
        self.mean_val = None

    def fit(self, X):
        self.mean_val = X.mean(axis=0)

    def predict(self, X):
        anomaly_score = 1 - X@self.mean_val
        return anomaly_score


if __name__ == '__main__':

    normal_data = np.loadtxt(
        "../input/normal_direction_data.csv", delimiter=",")
    error_data = np.loadtxt("../input/error_direction_data.csv", delimiter=",")

    model = DirectionalDataAnomalyDetection()
    model.fit(normal_data)
    pred = model.predict(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
