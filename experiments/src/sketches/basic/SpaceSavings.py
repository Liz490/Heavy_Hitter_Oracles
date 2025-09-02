from experiments.src.sketches.heapbased.heap import Heap
from sortedcontainers import SortedDict
import heapq
import numpy as np

class SpaceSavings:

    def __init__(self, S: int,):
        self.S = S
        self.heap = {}
        self.els = 0

    def update(self, x: int, d: int = 1) -> None:
        if x in self.heap: 
            self.heap[x] += d
        else:
            if self.els < self.S:
                self.heap[x] = d
                self.els += 1
            else:
                min_key = min(self.heap, key=self.heap.get)

                min_val = self.heap[min_key]
                del self.heap[min_key]
                self.heap[x] = d + min_val

    def estimate(self, x: int) -> int:
        """Estimate the frequency of x"""
        if x in self.heap:
            return self.heap[x]
        else:
            return 0
        
    def get_top_k(self, k: int) -> np.ndarray: 
        if k > len(self.heap):
            raise ValueError(f"Requested top {k} elements, but only {len(self.heap)} elements are available.")
        top_k = heapq.nlargest(k, self.heap, key=self.heap.get)
        return np.array([int(x) for x in top_k])
    
    def get_top_k_with_freqs(self, k: int) -> dict[int,int]:
        """Get the top k elements with their frequencies"""
        if k > len(self.heap):
            raise ValueError(f"Requested top {k} elements, but only {len(self.heap)} elements are available.")
        top_k = heapq.nlargest(k, self.heap.items(), key=lambda item: item[1])
        return {int(x): int(freq) for x, freq in top_k}