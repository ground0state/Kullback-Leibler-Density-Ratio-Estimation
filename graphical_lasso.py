import numpy as np
import sys
from sklearn.preprocessing import StandardScaler


class GraphicalLasso():
    def __init__(self):
        self.conv = None
        self.pmatrix = None
        self.pmatrix_inv = None
        self.best_score = None

        self.pmatrix_new = None
        self.pmatrix_inv_new = None
        self.conv_new = None
        self.best_score_new = None

    def fit(self, X, rho=0.01, normalize=True):
        self.pmatrix, self.pmatrix_inv, self.conv, self.best_score = self._solve(
            X, rho=rho, normalize=normalize)

    def _solve(self, X, rho=0.01, normalize=True):
        if normalize:
            sc = StandardScaler()
            X = sc.fit_transform(X)

        conv = np.cov(X, rowvar=0, bias=1)
        pmatrix = conv
        pmatrix_inv = conv
        best_score = -sys.float_info.max
        # while True:
        for cnt in range(10):
            for i in range(X.shape[1]):
                W = np.delete(pmatrix_inv, i, 0)
                W = np.delete(W, i, 1)
                W_diagonal_zero = W - np.diagflat(np.diag(W))

                s = conv[:, i]
                s = np.delete(s, i, axis=0)

                # solve beta
                beta = np.ones(W.shape[0]) * 10
                best_d = sys.float_info.max
                while True:
                    A = s - beta@W_diagonal_zero
                    for idx, a in enumerate(A):
                        if a > rho:
                            beta[idx] = (a - rho)/W[idx, idx]
                        elif a < -rho:
                            beta[idx] = (a + rho)/W[idx, idx]
                        else:
                            beta[idx] = 0

                    target = beta@W - s + rho * np.sign(beta)
                    d = np.sum(target)
                    if d < best_d:
                        best_d = d
                    else:
                        # converge
                        break

                        # update pmatrix_inv
                w = beta@W
                sigma = conv[i, i] + rho
                w_ = np.insert(w, i, sigma)
                pmatrix_inv[:, i] = w_
                pmatrix_inv[i, :] = w_

                # update pmatrix
                lam = 1 / (sigma - beta@W@beta)
                l = - lam * beta
                l_ = np.insert(l, i, lam)
                pmatrix[:, i] = l_
                pmatrix[i, :] = l_

            score = np.log(np.linalg.det(pmatrix)) - \
                np.trace(conv@pmatrix) - rho*np.sum(np.abs(pmatrix))
            if score > best_score:
                best_score = score

        return pmatrix, pmatrix_inv, conv, best_score

    def outlier_analysis(self, X):
        diag = np.diag(self.pmatrix)
        anomaly_score = []
        for x in X:
            a = np.log(2*np.pi/diag)/2 + (x@self.pmatrix)**2/(2*diag)
            anomaly_score.append(a)
        return np.array(anomaly_score)

    def anomaly_analysis(self, X, rho=0.01, normalize=True):
        self.pmatrix_new, self.pmatrix_inv_new, self.conv_new, self.best_score_new = self._solve(
            X, rho=rho, normalize=normalize)

        diag = np.diag(self.pmatrix).reshape(-1)
        diag_new = np.diag(self.pmatrix_new).reshape(-1)
        diag_S = np.diag(self.pmatrix@self.conv@self.pmatrix).reshape(-1)
        diag_S_new = np.diag(self.pmatrix_new@self.conv@
                             self.pmatrix_new).reshape(-1)

        a = np.log(diag/diag_new)/2 - (diag_S/diag - diag_S_new/diag_new)/2

        return a


if __name__ == '__main__':
    normal_data = np.loadtxt("./input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("./input/error_data.csv", delimiter=",")

    model = GraphicalLasso()
    model.fit(normal_data)
    pred = model.outlier_analysis(error_data)

    import matplotlib.pyplot as plt
    for k in range(pred.shape[1]):
        plt.plot(pred[:, k])
    plt.show()

    print(model.anomaly_analysis(error_data, rho=0.1))
