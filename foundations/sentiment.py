import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.network = nn.Sequential(
                    #Embedding layer 
                    nn.Embedding(vocabulary_size, 16),
                    nn.Linear(16,1),
                    nn.Sigmoid()
                )


    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        embeddings = self.network[0](x)
        # but you should average it into a B, embed_dim tensor before using the Linear layer   
        avg_embeddings = embeddings.mean(dim= 1)

        # Return a B, 1 tensor and round to 4 decimal places
                # Linear + Sigmoid
        output = self.network[1:](avg_embeddings)  # (B, 1)

        # Round to 4 decimal places
        return torch.round(output * 10000) / 10000
        
