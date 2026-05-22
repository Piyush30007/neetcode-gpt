import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        help = []
        z_max= np.max(z)
        for i in range(len(z)):
            help.append(np.exp(z[i]- z_max))
        sum = 0
        for i in range(len(help)):
            sum+=help[i]
        help = np.array(help)/sum
        return np.round(help,4)    