import matplotlib.pyplot as plt
import numpy as np
from pyanom.outlier_detection import HotelingT2


normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

model = HotelingT2()
model.fit(normal_data)
pred = model.score(error_data)

plt.plot(pred)
plt.show()
