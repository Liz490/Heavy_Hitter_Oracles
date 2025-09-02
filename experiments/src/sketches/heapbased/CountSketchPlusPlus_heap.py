import math
import numpy as np
from experiments.src.sketches.basic.CountSketchPlusPlus import CountSketchPlusPlus
from experiments.src.sketches.heapbased.heap import Heap

class CountSketchPlusPlusHeap(): 

    def __init__(self, n: int, S_total: int, seed:int, S_heap:int = 0):
        self.n = n
        self.S_total = S_total
        self.S_heap = S_heap
        self.S_sketch = S_total - S_heap
        self.tau_prime = math.ceil(math.log2(n))
        if S_heap > S_total:
            raise ValueError("S_heap cannot be larger than S_total")
        self.CountSketchPlusPlus = CountSketchPlusPlus(n, S= S_total - S_heap, seed = seed)
        self.heap = Heap(size=S_heap)


    def update(self, x: np.int64, d: int = 1) -> None:
        """Update the sketch and heap with d copies of x"""

        if x in self.heap: 
            self.heap.put(x, d)
        elif self.heap.num_els < self.S_heap:
            self.heap.put(x, d, 0)
            self.heap.num_els += 1
        else:
            f_x_small_table, f_x_big_table = self.CountSketchPlusPlus.get_small_and_big_table_estimates(x)
            if f_x_small_table > self.tau_prime: 
                min_key, min_val = self.heap.get_min()
                if f_x_big_table + d > min_val:
                    self.heap.put(x, f_x_big_table + d, f_x_big_table)
                    count_diff = min_val - self.heap.get_old_value(min_key)
                    self.heap.delete(min_key)
                    self.CountSketchPlusPlus.update(np.int64(min_key), count_diff)
                    return 
            self.CountSketchPlusPlus.update(x, d)
      
     
    def get_top_k(self, k: int) -> np.ndarray: 
        """Get the top k elements from the heap"""
        return self.heap.get_top_k(k)
    
    def get_top_k_with_freqs(self, k: int) -> dict[int,int]:
        """Get the top k elements with their frequencies from the heap"""
        return self.heap.get_top_k_with_freqs(k)


    def estimate(self, x: np.int64, tau: float) -> float:
        """Estimate the frequency of x"""

        if x in self.heap:
            return self.heap[x]
        else: 
            f_x_small_table, f_x_big_table = self.CountSketchPlusPlus.get_small_and_big_table_estimates(x)
            if f_x_small_table < tau:
                return 0
            else:
                return f_x_big_table
            
        

            

            
