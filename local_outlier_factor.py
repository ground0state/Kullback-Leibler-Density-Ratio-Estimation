if __name__ == '__main__':

    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    normal_data = np.loadtxt("./input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("./input/error_data.csv", delimiter=",")

    normal_data = StandardScaler().fit_transform(normal_data)
    error_data = StandardScaler().fit_transform(error_data)

    k = 5
    clf = LocalOutlierFactor(n_neighbors=k, novelty=True)
    clf.fit(normal_data)
    pred = clf.predict(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
