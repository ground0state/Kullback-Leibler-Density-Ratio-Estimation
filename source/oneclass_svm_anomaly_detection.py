if __name__ == '__main__':
    import numpy as np
    from sklearn.svm import OneClassSVM

    normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

    class Args():
        kernel = "rbf"
        degree = 3
        gamma = "auto"

    model = OneClassSVM(
        kernel=Args().kernel,
        degree=Args().degree,
        gamma=Args().gamma,
    ).fit(normal_data)

    y_pred = -np.log(model.score_samples(error_data))

    import matplotlib.pyplot as plt
    plt.plot(y_pred)
    plt.show()
