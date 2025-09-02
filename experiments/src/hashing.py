import xxhash
from functools import lru_cache
import numpy as np 


class HashFns:
    def __init__(self, seed):
        self.seed = seed

    @lru_cache(maxsize=330_000)
    def hash(self, item:np.int64 ):
        # Only use for positive integers
        item_bytes = item.tobytes()
        return xxhash.xxh64_intdigest(item_bytes, seed=self.seed)