import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:
        # first convert this into numpy arrays
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward
        z1 = np.dot(x, W1.T) + b1
        a1 = np.maximum(z1, 0)

        z2 = np.dot(a1, W2.T) + b2
        prediction = z2

        # Loss
        loss = np.mean((prediction - y_true) ** 2)

        # Backward
        dz2 = 2 * (prediction - y_true) / len(y_true)

        dW2 = np.outer(dz2, a1)
        db2 = dz2

        da1 = np.dot(dz2, W2)

        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }
