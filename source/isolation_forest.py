"""
Copyright © 2019 ground0state. All rights reserved.
"""
if __name__ == '__main__':

    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
    error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

    normal_data = StandardScaler().fit_transform(normal_data)
    error_data = StandardScaler().fit_transform(error_data)

    k = 5
    clf = IsolationForest(
        n_estimators=100, max_samples="auto",  max_features=1.0)
    clf.fit(normal_data)
    pred = clf.predict(error_data)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.show()
