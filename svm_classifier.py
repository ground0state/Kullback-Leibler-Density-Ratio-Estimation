if __name__ == '__main__':
    import numpy as np
    from sklearn.svm import SVC

    error_data = np.loadtxt(
        "./input/error_data_with_label.csv", delimiter=",")

    X_train = error_data[:50, 1:]
    y_train = error_data[:50, :1].ravel()
    X_valid = error_data[50:, 1:]
    y_valid = error_data[50:, :1].ravel()

    class Args():
        C = 0.1
        kernel = "rbf"
        degree = 3
        max_iter = -1

    svc = SVC(
        C=Args().C,
        kernel=Args().kernel,
        degree=Args().degree,
        max_iter=Args().max_iter
    ).fit(X_train, y_train)

    y_pred = svc.predict(X_valid)

    from sklearn.metrics import jaccard_score
    print("jaccard_score:", jaccard_score(y_valid, y_pred))

    import matplotlib.pyplot as plt
    plt.plot(y_pred)
    plt.show()
