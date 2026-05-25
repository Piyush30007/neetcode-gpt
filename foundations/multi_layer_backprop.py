import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = np.dot(x, np.transpose(W1)) + b1
        alpha1 = np.maximum(0, z1)
        z2 = np.dot(alpha1 , np.transpose(W2))+b2 
        loss = np.mean((z2-y_true)**2)
        dz2 = 2*(z2-y_true)
        dw2 = np.outer(dz2, alpha1)
        db2 = dz2 
        da1=np.dot(np.transpose(W2),dz2)
        dz1 = da1 * (z1 > 0)
        dw1 = np.outer(dz1, x)+0

        return {
                "loss": float(np.round(loss, 4)),
                "dW1": np.round(dw1, 4).tolist(),
                "db1": np.round(dz1, 4).tolist(),
                "dW2": np.round(dw2, 4).tolist(),
                "db2": np.round(db2, 4).tolist()
            }
