import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        result = np.exp(-z)
        # for i in range(len(result)):
        #     result[i] = 1 / (1+result[i])
        result = 1 / (1+result)
        return np.round(result,5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        # for i in range (len(z)):
        #     z[i] = max(0 , z[i])
        z = np.maximum(0, z)
        return z
