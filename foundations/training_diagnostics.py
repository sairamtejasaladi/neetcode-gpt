import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> List[Dict[str, float]]:

        stats = []

        def hook(module, inputs, output):

            mean = output.mean()
            std = output.std()

            # Check each neuron across the batch
            dead = (output <= 0).all(dim=0)
            dead_fraction = dead.float().mean()

            stats.append({
                "mean": round(mean.item(), 4),
                "std": round(std.item(), 4),
                "dead_fraction": round(dead_fraction.item(), 4)
            })

        hooks = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(
                    layer.register_forward_hook(hook)
                )

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        return stats


    def compute_gradient_stats(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> List[Dict[str, float]]:

        model.zero_grad()

        predictions = model(x)

        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, y)

        loss.backward()

        stats = []

        for layer in model.modules():

            if isinstance(layer, nn.Linear):

                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats


    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]]
    ) -> str:

        # 1. Dead neurons
        for stats in activation_stats:
            if stats["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for stats in gradient_stats:
            if stats["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradient in last layer
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation standard deviation
        for stats in activation_stats:

            if stats["std"] < 0.1:
                return "vanishing_gradients"

            if stats["std"] > 10.0:
                return "exploding_gradients"

        # 5. Everything looks okay
        return "healthy"