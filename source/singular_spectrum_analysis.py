"""
Copyright (c) 2019 ground0state. All rights reserved.
"""
import numpy as np


class singular_spectrum_analysis():
    def __init__(self):
        self.score = None

    def fit(self, y, window_size, trajectory_n, trajectory_pattern, test_n, test_pattern, lag=0):
        assert window_size < len(y) + 1
        assert trajectory_pattern <= window_size
        assert test_pattern <= window_size
        assert trajectory_n >= 1
        assert test_n >= 1
        assert 0 <= lag < len(y) - window_size - test_n - 1

        X = np.asarray([y[i:i+window_size]
                        for i in range(len(y) - window_size - 1)])

        anomaly_score = []
        for t in range(window_size+test_n+1, len(y) - lag):
            # trajectory matrix and test matrix at t
            X_t = X[t-trajectory_n-window_size:t-window_size].T
            Z_t = X[t-test_n+lag -
                    window_size:t-window_size+lag].T

            # SVD
            U, s, _ = np.linalg.svd(X_t)
            U = U[:, :trajectory_pattern]

            Q, _, _ = np.linalg.svd(Z_t)
            Q = Q[:, :test_pattern]

            UhQ = np.dot(U.T, Q)
            _, s, _ = np.linalg.svd(UhQ)

            a = 1 - s[0]
            # regularize
            if a < 10e-10:
                a = 0
            anomaly_score.append(a)

        self.score = np.array(anomaly_score)

    def get_score(self):
        return self.score


if __name__ == '__main__':
    error_data = np.loadtxt("../input/timeseries_error2.csv", delimiter=",")

    model = singular_spectrum_analysis()
    model.fit(error_data, window_size=50, trajectory_n=25,
              trajectory_pattern=3, test_n=25, test_pattern=2, lag=25)
    pred = model.get_score()

    import matplotlib.pyplot as plt
    plt.plot(error_data)
    plt.show()

    plt.plot(pred)
    plt.show()
