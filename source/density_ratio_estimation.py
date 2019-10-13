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

    def fit(self, X_normal, X_error):
        self.theta = np.ones(len(X_normal))
        self.J = []

        self.psi = np.asarray([self._gauss_kernel(x, X_normal)
                               for x in X_normal])
        self.psi_prime = np.asarray(
            [self._gauss_kernel(x, X_normal) for x in X_error])
        dJ_1 = self.psi_prime.sum(axis=0) / len(X_error)

        for i in range(self.num_iterations):
            # calculate J
            r = np.dot(self.psi, self.theta)
            r_prime = np.dot(self.psi_prime, self.theta)
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
        return -np.log(r_prime)


if __name__ == '__main__':
    normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

    train_normal_data = normal_data[50:]
    valid_normal_data = normal_data[:50]
    train_error_data = error_data[50:]
    valid_error_data = error_data[:50]

    model = DensityRatioEstimation()
    model.fit(train_normal_data, train_error_data)
    pred = model.predict(valid_normal_data, valid_error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
