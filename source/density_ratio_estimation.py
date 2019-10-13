import numpy as np


class DensityRatioEstimation():
    def __init__(self, band_width=1.0, learning_rate=0.1, num_iterations=100):
        self.band_width = band_width
        self.theta = None
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.J = None
        self.psi = None
        self.psi_prime = None
        self.eps = 10e-15

    def fit(self, X_normal, X_error):
        self.theta = np.ones(len(X_normal))
        self.J = []

        self.psi = np.asarray([self._gauss_kernel(x, X_normal)
                               for x in X_normal])
        self.psi_prime = np.asarray(
            [self._gauss_kernel(x, X_normal) for x in X_error])
        dJ_1 = self.psi_prime.sum(axis=0) / len(X_error)

        for _ in range(self.num_iterations):
            # calculate J
            r = np.dot(self.psi, self.theta)
            r = np.maximum(r, self.eps)
            r_prime = np.dot(self.psi_prime, self.theta)
            r_prime = np.maximum(r_prime, self.eps)
            self.J.append(np.sum(r_prime)/len(X_error) -
                          np.sum(np.log(r))/len(X_normal))

            # calculate gradient
            dJ = dJ_1 - (self.psi / r).sum(axis=0) / len(X_normal)
            self.theta -= self.learning_rate * dJ

    def _gauss_kernel(self, x, X):
        return np.exp(-np.sum((x - X)**2, axis=1)/(2*self.band_width**2))

    def get_score(self):
        return self.J

    def objective(self, X_normal, X_error):
        # calculate J
        psi = np.asarray([self._gauss_kernel(x, X_normal)
                          for x in X_normal])
        psi_prime = np.asarray([self._gauss_kernel(x, X_normal)
                                for x in X_error])
        r = np.dot(psi, self.theta)
        r_prime = np.dot(psi_prime, self.theta)
        J = np.sum(r_prime)/len(X_error) - np.sum(np.log(r))/len(X_normal)
        return J

    def predict(self, X_normal, X_error):
        psi_prime = np.asarray([self._gauss_kernel(x, X_normal)
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
    scores = []
    ks_score = {}
    for k in ks:
        for train_index, valid_index in kf_iter:
            train_normal_data = normal_data[train_index]
            valid_normal_data = normal_data[valid_index]
            train_error_data = error_data[train_index]
            valid_error_data = error_data[valid_index]

            model = DensityRatioEstimation(
                band_width=k, learning_rate=0.1, num_iterations=1000)
            model.fit(train_normal_data, train_error_data)
            scores.append(model.get_score())

        ks_score[k] = np.mean(scores)

    min_k = min(ks_score, key=ks_score.get)
    print('min k:', min_k)

    model = DensityRatioEstimation(
        band_width=min_k, learning_rate=0.1, num_iterations=1000)
    model.fit(normal_data, error_data)

    scores = model.get_score()
    pred = model.predict(normal_data, error_data)

    plt.plot(scores)
    plt.show()

    plt.plot(pred)
    plt.show()
