"""
Copyright (c) 2019 ground0state. All rights reserved.
"""
if __name__ == '__main__':
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    error_data = np.loadtxt(
        "../input/error_data_with_label.csv", delimiter=",")

    X_train = error_data[:50, 1:]
    y_train = error_data[:50, :1].ravel()
    X_valid = error_data[50:, 1:]
    y_valid = error_data[50:, :1].ravel()

    class Args():
        n_neighbors = 10
        p = 2

    knc = KNeighborsClassifier(
        n_neighbors=Args().n_neighbors,
        p=Args().p
    ).fit(X_train, y_train)

    y_pred = knc.predict(X_valid)

    from sklearn.metrics import jaccard_score
    print("jaccard_score:", jaccard_score(y_valid, y_pred))

    import matplotlib.pyplot as plt
    plt.plot(y_pred)
    plt.show()
