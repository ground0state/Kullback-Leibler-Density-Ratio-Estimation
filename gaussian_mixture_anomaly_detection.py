if __name__ == '__main__':
    import numpy as np
    from sklearn.mixture import GaussianMixture

    normal_data = np.loadtxt("./input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("./input/error_data.csv", delimiter=",")

    class Args():
        n_components = 3
        covariance_type = "spherical"
        random_state = 0
        max_iter = 100

    gmm = GaussianMixture(
        n_components=Args().n_components,
        covariance_type=Args().covariance_type,
        random_state=Args().random_state,
        max_iter=Args().max_iter
    ).fit(normal_data)

    pred = - gmm.score_samples(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
