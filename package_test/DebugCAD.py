import matplotlib.pyplot as plt
import numpy as np
from pyanom.outlier_detection import CAD


normal_data = np.loadtxt(
    "../input/timeseries_normal.csv", delimiter=",").reshape(-1, 1)
error_data = np.loadtxt(
    "../input/timeseries_error.csv", delimiter=",").reshape(-1, 1)

model = CAD()
model.fit(normal_data, threshold=3)
pred = model.score(error_data, cumsum_on=True)

plt.plot(pred)
plt.show()
