if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import make_friedman2
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process import kernels

    X_train = np.loadtxt("./input/GaussianProcess_X_train.csv",
                         delimiter=",").reshape(-1, 1)
    y_train = np.loadtxt("./input/GaussianProcess_y_train.csv", delimiter=",")
    X_test = np.loadtxt("./input/GaussianProcess_X_test.csv",
                        delimiter=",").reshape(-1, 1)
    y_test = np.loadtxt("./input/GaussianProcess_y_test.csv", delimiter=",")

    kernel = kernels.RBF(1.0, (1e-3, 1e3)) + \
        kernels.ConstantKernel(1.0, (1e-3, 1e3)) + kernels.WhiteKernel()
    clf = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=20,
        normalize_y=True).fit(X_train, y_train)
    pred_mean, pred_std = clf.predict(X_test, return_std=True)

    def anomaly_score(pred_mean, pred_std, y_test):
        a = np.log(2*np.pi*pred_std**2)/2 + \
            (y_test.reshape(-1) - pred_mean)**2/(2*pred_std**2)
        return a

    a = anomaly_score(pred_mean, pred_std, y_test)

    import matplotlib.pyplot as plt
    plt.plot(X_test, a)
    plt.show()
