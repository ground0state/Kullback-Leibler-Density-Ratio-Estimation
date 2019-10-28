import matplotlib.pyplot as plt
import numpy as np
from pyanom.subspace_methods import SSA


error_data = np.loadtxt("../input/timeseries_error2.csv", delimiter=",")

model = SSA()
model.fit(error_data, window_size=50, trajectory_n=25,
          trajectory_pattern=3, test_n=25, test_pattern=2, lag=25)
pred = model.score()

plt.plot(error_data)
plt.show()

plt.plot(pred)
plt.show()
