"""
Copyright © 2019 ground0state. All rights reserved.
"""
import numpy as np


class CUMSUM():
    def __init__(self):
        self.normal_mean = None
        self.normal_std = None
        self.error_mean = None
        self.nu = None
        self.uppper = None

    def fit(self, y, threshold):
        self.normal_mean = np.mean(y)
        self.normal_std = np.std(y)
        self.error_mean = threshold
        self.nu = self.error_mean - self.normal_mean

        if self.nu > 0:
            self.uppper = True
        else:
            self.uppper = False

    def predict(self, y_test, cumsum_on=True):

        if self.uppper:
            anomaly_socre = self.nu * \
                (y_test - self.normal_mean - self.nu/2)/self.normal_std**2
        else:
            anomaly_socre = -1*self.nu * \
                (y_test - self.normal_mean + self.nu/2)/self.normal_std**2

        a_operated = 0
        anomaly_socre_cumsum = []
        for a in anomaly_socre:
            a += a_operated
            a_operated = np.maximum(a, 0)
            anomaly_socre_cumsum.append(a_operated)
        anomaly_socre_cumsum = np.array(anomaly_socre_cumsum)

        if cumsum_on:
            return anomaly_socre_cumsum
        else:
            return anomaly_socre


if __name__ == '__main__':
    normal_data = np.loadtxt("../input/timeseries_normal.csv", delimiter=",")
    error_data = np.loadtxt("../input/timeseries_error.csv", delimiter=",")

    model = CUMSUM()
    model.fit(normal_data, threshold=3)
    pred = model.predict(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
