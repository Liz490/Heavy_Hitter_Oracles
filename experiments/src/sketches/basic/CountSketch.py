
from experiments.src.hashing import HashFns
import numpy as np 
import math 
import bottleneck as bn
from functools import lru_cache
from experiments.src.sketches.abstract_sketch import Sketch
# Taken and adapted from supplementary material on https://openreview.net/forum?id=0VcvYQ3uPh&noteId=jVEqh6rQkg

class CountSketch(Sketch):
    def __init__(self, n, W, L, preds, seed):
        '''
        Initialize a CountSketch table with L rows and W columns
        '''
        np.random.seed(seed)
        self.n = n
        self.W = W  # number of buckets
        self.L = L  # number of rows
        self.rows = np.arange(L)
        self.rowHashes = [HashFns(seed=seed) for i in range(self.L)]
        self.signHashes = [HashFns(seed=seed) for i in range(self.L)]
        self.sketch = np.zeros((self.L, self.W), dtype=np.int64)
        self.preds = preds

        self.sketchMats = [np.zeros((self.W, self.n), dtype=np.int64) for j in range(self.L)]
        for i in range(n):
            i = np.int64(i) 
            for j in range(self.L): 
                row = self.rowHashes[j].hash(i) % self.W
                sign = 1 - 2 * (self.signHashes[j].hash(i) % 2)
                self.sketchMats[j][row, i] = sign

    @lru_cache(maxsize=330_000)
    def __ids(self, x: np.int64):
        cols = np.array([hash.hash(x) % self.W for hash in self.rowHashes])
        return self.rows, cols

    @lru_cache(maxsize=330_000)
    def __signs(self, x: np.int64):
        return 1 - (2 * np.array([hash.hash(x) % 2 for hash in self.signHashes]))

    def vectorUpdate(self, freqs: np.ndarray):
        '''Update sketch with a vector (frequency histogram)'''
        assert(len(freqs) == self.n)
        freqs[list(self.preds.keys())] = 0  # Ignore predictions in the update
        for l in range(self.L):
            self.sketch[l,:] += ((self.sketchMats[l] @ freqs.T).T).astype(np.int64)


    def update(self, x: np.int64, d=1):
        '''Update sketch with d copies of x'''
        if x in self.preds:
            return
        rows, cols = self.__ids(x)
        signs = self.__signs(x)
        self.sketch[rows, cols] += (signs * d).astype(np.int64)


    def getSketch(self):
        return self.sketch

    def estimate(self, x: np.int64, tau=None):
        if x in self.preds:
            return self.preds[x]
        rows, cols = self.__ids(x)
        signs = self.__signs(x)
        median = bn.median(self.sketch[rows, cols] * signs)
        if tau and median < tau: 
            return 0
        return median

    
    def run_and_evaluate_sketch(self, data: np.ndarray, estFuns):
        '''
        Evaluate a sketch on a given dataset which is a 
        vector of counts (the frequency histogram)
        '''
        uerrs = np.zeros(len(estFuns))
        werrs = np.zeros(len(estFuns))
        self.vectorUpdate(data)
        f_prime = [self.estimate(np.int64(i)) for i in range(self.n)]
        for j, estFun in enumerate(estFuns):
            final_estimates = np.array(estFun(f_prime))
            abs_errors = np.abs(final_estimates - data)
            uerr = bn.nanmean(abs_errors)
            werr = bn.nanmean(abs_errors * data)
            uerrs[j] = uerr
            werrs[j] = werr
        return uerrs, werrs
