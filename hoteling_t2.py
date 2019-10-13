import numpy as np
from scipy.stats import f


class HotelingT2():
    def __init__(self):
        self.mean_val = None
        self.cov_val_inv = None
        self.M = None
        self.N = None

    def fit(self, X):
        self.N, self.M = X.shape
        self.mean_val = X.mean(axis=0)
        self.cov_val_inv = np.linalg.inv(np.cov(X, rowvar=0, bias=1))

    def predict(self, X):
        pred = []
        for x in X:

            a = (x-self.mean_val)@self.cov_val_inv@(x-self.mean_val)
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

#     def predict(self, X):
#         pred = []
#         for x in X:
#             b = np.linalg.solve(self.cov_val, x-self.mean_val)
#             a = (x-self.mean_val)@b
#             # T2 = (self.N - self.M)/((self.N + 1) * self.M) * a
#             # prob = f.pdf(T2, self.M, self.N-self.M)
#             pred.append(a)

#         return np.asarray(pred)


if __name__ == '__main__':

    normal_data = np.loadtxt("./input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("./input/error_data.csv", delimiter=",")

    model = HotelingT2()
    model.fit(normal_data)
    pred = model.predict(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
