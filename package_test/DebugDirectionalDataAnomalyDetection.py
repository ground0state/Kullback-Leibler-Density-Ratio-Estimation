import matplotlib.pyplot as plt
import numpy as np
from pyanom.outlier_detection import DirectionalDataAnomalyDetection

normal_data = np.loadtxt(
    "../input/normal_direction_data.csv", delimiter=",")
error_data = np.loadtxt("../input/error_direction_data.csv", delimiter=",")

model = DirectionalDataAnomalyDetection()
model.fit(normal_data, normalize=True)
pred = model.score(error_data)

plt.plot(pred)
plt.show()
