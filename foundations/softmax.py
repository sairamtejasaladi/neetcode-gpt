import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum_ = 0 
        max_val = max(z)
        for i in z : 
            sum_ += np.exp(i-max_val)
        res = []
        for i in z : 
            res.append( np.exp(i-max_val)/sum_)
        return (np.round(res,4))
