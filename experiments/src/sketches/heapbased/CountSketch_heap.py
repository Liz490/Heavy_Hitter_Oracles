import numpy as np 
import bottleneck as bn
from experiments.src.sketches.basic.CountSketch import CountSketch
from experiments.src.sketches.heapbased.heap import Heap

class CountSketchHeap():
    """
    CountSketch with a heap to store the most frequent elements.
    The heap is used to store elements that exceed a certain threshold.
    """

    def __init__(self, n: int, S_total: int, seed:int, non_neg: bool = True, L: int = 3, S_heap: int = 0):
        self.heap = Heap(size=S_heap)
        self.S_heap = S_heap
        self.S_total = S_total
        self.S_sketch = S_total - S_heap
        if S_heap > S_total:
            raise ValueError("S_heap cannot be larger than S_total")
        self.sketch = CountSketch(n, int((S_total - S_heap)/L), L, preds={}, seed = seed)
        self.non_neg = non_neg


    # vectorize estimation function and update those values later, that were in the heap 
    def estimate(self, x: np.int64) -> int:
        """Estimate the frequency of x"""
        if x in self.heap:
            return self.heap[x]
        else:
            estimate = self.sketch.estimate(x)
            if self.non_neg: 
                return max(estimate, 0)
            else:
                return estimate

    def get_top_k(self, k: int) -> np.ndarray: 
        """Get the top k elements from the heap"""
        return self.heap.get_top_k(k)

    def get_top_k_with_freqs(self, k: int) -> dict[int,int]:
        """Get the top k elements with their frequencies from the heap"""
        return self.heap.get_top_k_with_freqs(k)


    def update(self, x: np.int64, d: int = 1) -> None:
        """Update the sketch and heap with d copies of x"""
        
        if x in self.heap: 
            self.heap.put(x, d)
        elif self.heap.num_els < self.S_heap:
            self.heap.put(x, d, 0)
            self.heap.num_els += 1
        else:
            f_x = self.estimate(x)
            min_key, min_val = self.heap.get_min()
            if f_x + d > min_val:
                self.heap.put(x, f_x + d, f_x)
                count_diff = min_val - self.heap.get_old_value(min_key)
                self.heap.delete(min_key)
                self.sketch.update(np.int64(min_key), count_diff)
            else:  
                self.sketch.update(np.int64(x), d)

    def evaluate(self, freq_vector) -> tuple[float, float]:
        """Evaluate the algorithm"""
        uerrs = []
        werrs = []
        for i, f_i in enumerate(freq_vector):
            est = self.estimate(np.int64(i))
            werrs.append(abs(est - f_i) * f_i)
            uerrs.append(abs(est - f_i))
        return bn.nanmean(uerrs), bn.nanmean(werrs)
               
        