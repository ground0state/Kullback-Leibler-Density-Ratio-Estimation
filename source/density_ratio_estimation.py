"""
Copyright (c) 2019 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np


class KLDensityRatioEstimation():
    """Kullback-Leibler density ratio estimation.

    Parameters
    ----------
    band_width : float
        Smoothing parameter gaussian kernel.

    learning_rate: float
        Learning rate.

    num_iterations: int
        Number of iterations over the training dataset to perform training.
    """

    def __init__(self, band_width=1.0, learning_rate=0.1, num_iterations=100):
        self.band_width = band_width
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.Js = None
        self.psi = None
        self.psi_prime = None
        self.eps = 10e-15

    def fit(self, X_normal, X_error):
        """Fit the DensityRatioEstimation model according to the given training data.

        Parameters
        ----------
        X_normal : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object

        Notes
        -----
        Use X_normal for basic function.
        """

        self.theta = np.ones(len(X_normal))
        self.Js = []
        self.psi = np.asarray([self._gaussian_kernel(x, X_normal)
                               for x in X_normal])
        self.psi_prime = np.asarray(
            [self._gaussian_kernel(x, X_normal) for x in X_error])
        dJ_1 = self.psi_prime.sum(axis=0) / len(X_error)

        for _ in range(self.num_iterations):
            # calculate J
            r = np.dot(self.psi, self.theta)
            r = np.maximum(r, self.eps)
            r_prime = np.dot(self.psi_prime, self.theta)
            r_prime = np.maximum(r_prime, self.eps)
            J = np.sum(r_prime)/len(X_error) - np.sum(np.log(r))/len(X_normal)
            self.Js.append(J)

            # calculate gradient
            dJ = dJ_1 - (self.psi / r).sum(axis=0) / len(X_normal)
            self.theta -= self.learning_rate * dJ
        self.Js = np.array(self.Js)

        return self

    def _gaussian_kernel(self, x, X):
        return np.exp(-np.sum((x - X)**2, axis=1)/(2*self.band_width**2))

    def get_running_loss(self):
        """Kullback-Leibler density ratio estimation.

        Returns
        -------
        Js : array-like, shape (num_iterations,)
            losses of objective function in training.
        """
        return self.Js

    def score(self, X_normal, X_error):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X_normal : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """

        psi_prime = np.asarray([self._gaussian_kernel(x, X_normal)
                                for x in X_error])
        r_prime = np.dot(psi_prime, self.theta)
        r_prime = np.maximum(r_prime, self.eps)
        return -np.log(r_prime)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold

    normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

    kf_iter = KFold(n_splits=3).split(normal_data)

    # A rule-of-thumb bandwidth estimator
    # <https://en.wikipedia.org/wiki/Kernel_density_estimation>
    SILVERMAN = 1.06*np.std(normal_data, axis=0)/pow(len(normal_data), 1/5)

    ks = SILVERMAN + [0.1, 0.5, 1.0]
    losses = []
    ks_loss = {}
    for k in ks:
        for train_index, valid_index in kf_iter:
            train_normal_data = normal_data[train_index]
            valid_normal_data = normal_data[valid_index]
            train_error_data = error_data[train_index]
            valid_error_data = error_data[valid_index]

            model = KLDensityRatioEstimation(
                band_width=k, learning_rate=0.1, num_iterations=1000)
            model.fit(train_normal_data, train_error_data)
            losses.append(model.get_running_loss()[-1])

        ks_loss[k] = np.mean(losses)

    min_k = min(ks_loss, key=ks_loss.get)
    print('min k:', min_k)

    model = KLDensityRatioEstimation(
        band_width=min_k, learning_rate=0.1, num_iterations=1000)
    model.fit(normal_data, error_data)

    losses = model.get_running_loss()
    pred = model.score(normal_data, error_data)

    plt.plot(losses)
    plt.show()

    plt.plot(pred)
    plt.show()
