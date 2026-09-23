import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_arr = np.array(x)
        x_rms = np.sqrt((np.sum(x_arr * x_arr))/len(x) + eps)
        x_norm = x_arr/x_rms
        res = list(np.round((gamma*x_norm), 4))
        return res
