import heapq
import numpy as np

class MisraGries():
    def __init__(self, S: int):
        self.S = S
        self.counters = {}
        self.last_deleted_keys = []
        self.num_els = 0

    def update(self, x: int, d: int = 1) -> None:
        """Update the sketch with the value i"""
        if x in self.counters:
            self.counters[x] += d
        elif self.num_els < self.S:
            self.counters[x] = d
            self.last_deleted_keys.pop() if self.last_deleted_keys else None
            self.num_els += 1
        else:
            keys_to_remove = []
            for key in self.counters.keys():
                self.counters[key] -= d
                if self.counters[key] <= 0:
                    keys_to_remove.append(key)
                    self.num_els -= 1
            for key in keys_to_remove:
                del self.counters[key]
            self.last_deleted_keys = keys_to_remove
            

    def estimate(self, x: int) -> int:
        """Estimate the frequency of i"""
        return self.counters.get(x, 0)
    
    def get_top_k(self, k: int) -> np.ndarray: 
        if len(self.counters) + len(self.last_deleted_keys) < k:
            raise ValueError(f"Requested top {k} elements, but only {len(self.counters)} elements are available.")
        top_k = heapq.nlargest(k, self.counters, key=self.counters.get)
        missing_els = k - len(top_k)
        if missing_els > 0:
            top_k += self.last_deleted_keys[:missing_els]
        return np.array([int(x) for x in top_k])
    
    def get_top_k_with_freqs(self, k: int) -> dict[int,int]:
        """Get the top k elements with their frequencies"""
        if len(self.counters) + len(self.last_deleted_keys) < k:
            raise ValueError(f"Requested top {k} elements, but only {len(self.counters)} elements are available.")
        top_k = heapq.nlargest(k, self.counters.items(), key=lambda item: item[1])
        missing_els = k - len(top_k)
        if missing_els > 0:
            top_k += [(x, 0) for x in self.last_deleted_keys[:missing_els]]
        return {int(x): int(freq) for x, freq in top_k}