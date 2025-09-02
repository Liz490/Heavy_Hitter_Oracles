import random
from sortedcontainers import SortedDict
import numpy as np
class Heap(): 
    
    def __init__(self, size: int):
        self.key_to_value = {}
        self.value_to_keys = SortedDict()
        self.key_to_old_value = {}
        self.size = size
        self.num_els = 0

    def __getitem__(self, key):
        """Get the value associated with the key in the heap."""
        if key not in self.key_to_value:
            raise KeyError(f"Key {key} not found in heap.")
        return self.key_to_value[key]
    
    def get_old_value(self, key):
        """Get the old value associated with the key in the heap."""
        if key not in self.key_to_old_value:
            raise KeyError(f"Key {key} not found in heap.")
        return self.key_to_old_value[key]

    def __contains__(self, key):
        """Check if the key is in the heap."""
        return key in self.key_to_value

    def put(self, key, value, old_sketch_value=None):
        if key in self.key_to_value:
            old_value = self.key_to_value[key]
            self.value_to_keys[old_value].remove(key)
            if not self.value_to_keys[old_value]:
                del self.value_to_keys[old_value]
        self.key_to_value[key] = value
        if value not in self.value_to_keys or self.value_to_keys[value] is None:
            self.value_to_keys[value] = set()
        self.value_to_keys[value].add(key)
        if old_sketch_value is not None:
            self.key_to_old_value[key] = old_sketch_value


    def delete(self, key) : 
        if key not in self.key_to_value:
            raise KeyError(f"Key {key} not found in heap.")
        value = self.key_to_value[key]
        self.value_to_keys[value].remove(key)
        if not self.value_to_keys[value]:
            del self.value_to_keys[value]
        del self.key_to_value[key]
        del self.key_to_old_value[key]

    def get_min(self) -> tuple[int, float]:
        min_val = self.value_to_keys.peekitem(0)[0]
        all_min_key = self.value_to_keys.peekitem(0)[1]
        key = random.choice(list(all_min_key))
        return key, min_val
    
    def get_top_k(self, k) -> np.ndarray:
        if k > len(self.key_to_value):
            raise ValueError(f"Requested top {k} elements, but only {len(self.key_to_value)} elements are available.") 
        if not self.value_to_keys:
            raise RuntimeError("Heap is empty. You need to run the algorithm on a datastream before retrieving the top k elements")
        top_k = list(self.key_to_value.keys())[:k]
        return np.array([int(x) for x in top_k])
    
    def get_top_k_with_freqs(self, k) -> dict[int,int]:
        """Get the top k elements with their frequencies"""
        if k > len(self.key_to_value):
            raise ValueError(f"Requested top {k} elements, but only {len(self.key_to_value)} elements are available.")
        if not self.value_to_keys:
            raise RuntimeError("Heap is empty. You need to run the algorithm on a datastream before retrieving the top k elements")
        top_k = list(self.key_to_value.items())[:k]
        return {int(x): int(freq) for x, freq in top_k}

    def __len__(self):
        return len(self.key_to_value)
