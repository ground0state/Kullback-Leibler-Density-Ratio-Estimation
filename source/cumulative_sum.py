"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np


class CAD():
    """CUSUM Anomaly Detection.
    """

    def __init__(self):
        self.normal_mean = None
        self.normal_std = None
        self.error_mean = None
        self.nu = None
        self.uppper = None

    def fit(self, y, threshold):
        """Fit the model according to the given train data.

        Parameters
        ----------
        y : array-like, shape (n_samples, )
            Normal measured vectors, where n_samples is the number of samples.

        threshold: float
            Size of the shift that is to be detected.

        Returns
        -------
        self : object
        """

        self.normal_mean = np.mean(y)
        self.normal_std = np.std(y)
        self.error_mean = threshold
        self.nu = self.error_mean - self.normal_mean

        if self.nu > 0:
            self.uppper = True
        else:
            self.uppper = False

        return self

    def score(self, y_test, cumsum_on=True):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        y_test : array-like, shape (n_samples,)
            Error measured vectors, where n_samples is the number of samples.

        cumsum_on: bool
            If True, return cumsumed anomaly score. If False, return pure anomaly score. 

        Returns
        -------
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """

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
    normal_data = np.loadtxt(
        "../input/timeseries_normal.csv", delimiter=",").reshape(-1, 1)
    error_data = np.loadtxt(
        "../input/timeseries_error.csv", delimiter=",").reshape(-1, 1)

    model = CAD()
    model.fit(normal_data, threshold=3)
    pred = model.score(error_data, cumsum_on=True)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
