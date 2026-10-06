import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        res = []
        def hook(module, input , output):
            dead = (output == 0).all(dim=0)
            dead_fr = dead.float().mean()
            res.append(round(dead_fr.item(), 4))

        hooks = []
        for layer in model.modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(hook))
        with torch.no_grad():
            model(x) # run the model
        for h in hooks : 
            h.remove()
        return res

    def suggest_fix(self, dead_fractions: List[float]) -> str:

        # 1. Any layer has > 50% dead neurons
        for df in dead_fractions:
            if df > 0.5:
                return 'use_leaky_relu'

        # 2. First layer has > 30% dead neurons
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. Dead fraction strictly increases with depth
        #    AND last layer > 0.1
        increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )

        if increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # 4. Maximum dead fraction < 0.1
        if max(dead_fractions) < 0.1:
            return 'healthy'

        # 5. Otherwise
        return 'healthy'
        

