import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_positional_encoding(
        self, seq_len: int, d_model: int
    ) -> NDArray[np.float64]:

        pos = np.arange(seq_len)[:, np.newaxis]
        print(pos)
        i = np.arange(0, d_model, 2)

        denominator = np.exp(i * (-np.log(10000.0) / d_model))

        angles = pos * denominator

        pe = np.zeros((seq_len, d_model))

        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])

        return np.round(pe, 5)