import matplotlib.pyplot as plt
from density_ratio_estimation import DensityRatioEstimation
import numpy as np
from sklearn.model_selection import KFold

normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

kf_iter = KFold(n_splits=3).split(normal_data)
ks = [1.0, 2.0, 3.0]
scores = []
ks_score = {}
for k in ks:
    for train_index, valid_index in kf_iter:
        train_normal_data = normal_data[train_index]
        valid_normal_data = normal_data[valid_index]
        train_error_data = error_data[train_index]
        valid_error_data = error_data[valid_index]

        model = DensityRatioEstimation(band_width=k)
        model.fit(train_normal_data, train_error_data)
        scores.append(model.get_score())

    ks_score[k] = np.mean(scores)

min_k = min(ks_score, key=ks_score.get)
print('min k:', min_k)

model = DensityRatioEstimation(band_width=min_k)
model.fit(normal_data, error_data)

pred = model.predict(normal_data, error_data)

plt.plot(pred)
plt.show()
