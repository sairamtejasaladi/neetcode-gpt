from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters/tokens
        tokens: List[str] = list(corpus)
        merges: List[List[str]] = []

        # 2. Perform up to `num_merges` merge steps
        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # 2a. Count frequency of all adjacent token pairs
            freq: dict[tuple[str, str], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                freq[pair] = freq.get(pair, 0) + 1

            if not freq:
                break

            # 2b. Most frequent pair; break ties lexicographically (smallest pair first)
            best_pair = min(freq.keys(), key=lambda p: (-freq[p], p))
            a, b = best_pair

            # 2c. Merge all non-overlapping occurrences left to right
            merged_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged_tokens.append(a + b)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            tokens = merged_tokens

            # 2d. Record the merge as [token_a, token_b]
            merges.append([a, b])

        # 3. Return the list of merges performed
        return merges