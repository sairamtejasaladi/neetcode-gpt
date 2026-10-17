import torch
import torch.nn as nn
import math
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(K.shape[-1])

        seq_len = embedded.shape[1]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=embedded.device)
        )
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ V

        return torch.round(output * 10000) / 10000