import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        if training : 
            x_ar = np.array(x)
            x_mean = np.mean(x , axis = 0)  
            x_var = np.var(x , axis =0)
            x_hat = (x_ar - x_mean)/ np.sqrt(x_var + eps)

            y = gamma * x_hat + beta

            running_mean = (1-momentum)*np.array(running_mean) + momentum * x_mean 
            running_var = (1-momentum)*np.array(running_var) + momentum * x_var
        
            return (list(np.round(y , 4)), list(np.round(running_mean , 4)), list(np.round(running_var, 4)))
        else : 
            x_ar = np.array(x)

            x_hat = (x_ar - np.array(running_mean))/ np.sqrt(np.array(running_var) + eps)

            y = gamma * x_hat + beta
        
            return (list(np.round(y , 4)), list(np.round(running_mean , 4)), list(np.round(running_var, 4)))





