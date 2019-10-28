"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
# from scipy.stats import f


class HotelingT2():
    """Hotelling's t-squared statistic.
    """

    def __init__(self):
        self.mean_val = None
        self.cov_val_inv = None
        self.M = None
        self.N = None

    def fit(self, X):
        """Fit the Hotelling's t-squared model according to the given train data.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            Normal measured vectors, where n_samples is the number of samples.

        Returns
        -------
        self : object
        """
        self.N, self.M = X.shape
        self.mean_val = X.mean(axis=0)
        if self.M > 1:
            self.cov_val_inv = np.linalg.inv(np.cov(X, rowvar=0, bias=1))
        elif self.M == 1:
            self.cov_val_inv = 1/np.var(X)
        else:
            raise ValueError("Input shape is incorrect")

        return self

    def score(self, X):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Error measured vectors, where n_samples is the number of samples.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """
        pred = []
        for x in X:
            if self.M > 1:
                a = (x-self.mean_val)@self.cov_val_inv@(x-self.mean_val)
            elif self.M == 1:
                a = (x-self.mean_val)**2*self.cov_val_inv
            # T2 = (self.N - self.M)/((self.N + 1) * self.M) * a
            # prob = f.pdf(T2, self.M, self.N-self.M)
            pred.append(a)

        return np.asarray(pred)

# slowly
# class HotelingT2():
#     def __init__(self):
#         self.mean_val = None
#         self.cov_val = None
#         self.M = None
#         self.N = None

#     def fit(self, X):
#         self.N, self.M = X.shape
#         self.mean_val = X.mean(axis=0)
#         self.cov_val = np.cov(X, rowvar=0, bias=1)

#     def score(self, X):
#         pred = []
#         for x in X:
#             b = np.linalg.solve(self.cov_val, x-self.mean_val)
#             a = (x-self.mean_val)@b
#             # T2 = (self.N - self.M)/((self.N + 1) * self.M) * a
#             # prob = f.pdf(T2, self.M, self.N-self.M)
#             pred.append(a)

#         return np.asarray(pred)


if __name__ == '__main__':

    normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

    model = HotelingT2()
    model.fit(normal_data)
    pred = model.score(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
