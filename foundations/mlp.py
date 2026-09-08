import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

        # no of hidden layers 
        n = len(weights)
        i = 0 
        for i in range(0,n-1):
        # layer weights 
            l1_w = np.array(weights[i])
        #hidden_layer_1 
            h1 = np.dot(l1_w.T,x)+np.array(biases[i])
        #relu 
            rel_ = np.maximum(h1,0)
            x = rel_
            i += 1 
        # final layer weights 
        w_f = np.array(weights[n-1])
        h2 = np.dot(w_f.T , x) + np.array(biases[n-1])

        return np.round(h2, 5)
