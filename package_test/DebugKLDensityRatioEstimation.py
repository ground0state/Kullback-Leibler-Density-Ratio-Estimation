import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from pyanom.density_ratio_estimation import KLDensityRatioEstimation
import numpy as np

normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

kf_iter = KFold(n_splits=3).split(normal_data)

# A rule-of-thumb bandwidth estimator
# <https://en.wikipedia.org/wiki/Kernel_density_estimation>
SILVERMAN = 1.06*np.std(normal_data, axis=0)/pow(len(normal_data), 1/5)

ks = SILVERMAN + [0.1, 0.5, 1.0]
scores = []
ks_score = {}
for k in ks:
    for train_index, valid_index in kf_iter:
        train_normal_data = normal_data[train_index]
        valid_normal_data = normal_data[valid_index]
        train_error_data = error_data[train_index]
        valid_error_data = error_data[valid_index]

        model = KLDensityRatioEstimation(
            band_width=k, learning_rate=0.1, num_iterations=1000)
        model.fit(train_normal_data, train_error_data)
        scores.append(model.get_running_loss()[-1])

    ks_score[k] = np.mean(scores)

min_k = min(ks_score, key=ks_score.get)
print('min k:', min_k)

model = KLDensityRatioEstimation(
    band_width=min_k, learning_rate=0.1, num_iterations=1000)
model.fit(normal_data, error_data)

scores = model.get_running_loss()
pred = model.score(normal_data, error_data)

plt.plot(scores)
plt.show()

plt.plot(pred)
plt.show()
