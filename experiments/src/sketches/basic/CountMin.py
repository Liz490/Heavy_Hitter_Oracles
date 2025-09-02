from experiments.src.hashing import HashFns
import numpy as np 
from functools import lru_cache
import bottleneck as bn
from experiments.src.sketches.abstract_sketch import Sketch

class CountMin(Sketch):
    def __init__(self, n, W, L, preds, seed):
        '''
        Initialize a CountMin Sketch table with L rows and W columns
        '''
        np.random.seed(seed)
        self.n = n
        self.W = W  # number of buckets
        self.L = L  # number of rows
        self.hashes = [HashFns(seed=seed) for i in range(self.L)] 
        self.sketch = np.zeros((self.L, self.W), dtype=np.int64)
        self.preds = preds

        self.sketchMats = [np.zeros((self.W, self.n), dtype=np.int64) for j in range(self.L)]
        for i in range(n):
            i = np.int64(i)  # Ensure item is an integer
            for j in range(self.L): 
                row = self.hashes[j].hash(i) % self.W
                self.sketchMats[j][row, i] = 1

    @lru_cache(maxsize=330_000)
    def __ids(self, x):
        cols = np.array([hash.hash(x) % self.W for hash in self.hashes])
        rows = np.arange(self.L)
        return rows, cols

    def vectorUpdate(self, freqs):
        '''Update sketch with a vector (frequency histogram)'''
        freqs[list(self.preds.keys())] = 0  # Ignore predictions in the update
        assert(len(freqs) == self.n)
        for l in range(self.L):
            self.sketch[l,:] += ((self.sketchMats[l] @ freqs.T).T).astype(np.int64)

    def update(self, x: np.int64, d=1):
        '''Update sketch with d copies of x'''
        if x in self.preds:
            return
        rows, cols = self.__ids(x)
        self.sketch[rows, cols] += d

    def getSketch(self):
        return self.sketch

    def estimate(self, x: np.int64):
        if x in self.preds:
            return self.preds[x]
        rows, cols = self.__ids(x)
        return bn.nanmin(self.sketch[rows, cols])

    
    def run_and_evaluate_sketch(self, data: np.ndarray,):
        '''
        Evaluate a sketch on a given dataset which is a 
        vector of counts (the frequency histogram)
        '''
        self.vectorUpdate(data)
        estimates = [self.estimate(np.int64(i)) for i in range(self.n)]
        abs_errors = np.abs(estimates - data)
        uerr = bn.nanmean(abs_errors)
        werr = bn.nanmean(abs_errors * data)
        return uerr, werr
