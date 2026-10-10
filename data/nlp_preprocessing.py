import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(
        self, positive: List[str], negative: List[str]
    ) -> TensorType[float]:

        # 1. Build vocabulary from unique words and sort them
        all_words = sorted(set(
            word
            for sentence in positive + negative
            for word in sentence.split()
        ))

        # 2. Assign IDs starting from 1 (0 is reserved for padding)
        vocab = {word: i + 1 for i, word in enumerate(all_words)}

        # 3. Encode each sentence as a tensor of word IDs
        encoded = []
        for sentence in positive + negative:
            ids = [vocab[word] for word in sentence.split()]
            encoded.append(torch.tensor(ids, dtype=torch.long))

        # 4. Pad shorter sequences with zeros
        result = nn.utils.rnn.pad_sequence(
            encoded, batch_first=True, padding_value=0
        )

        return result