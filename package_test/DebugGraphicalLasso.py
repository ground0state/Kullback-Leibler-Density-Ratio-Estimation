import matplotlib.pyplot as plt
import numpy as np
from pyanom.structure_learning import GraphicalLasso

normal_data = np.loadtxt("../input/normal_data.csv", delimiter=",")
error_data = np.loadtxt("../input/error_data.csv", delimiter=",")

model = GraphicalLasso()
model.fit(normal_data)
pred = model.outlier_analysis_score(error_data)

for k in range(pred.shape[1]):
    plt.plot(pred[:, k])
plt.show()

anomaly_score, precision_matrix = model.anomaly_analysis_score(
    error_data, rho=0.1)

print(anomaly_score)
print(precision_matrix, precision_matrix)
