
import math
import numpy as np
from collections import defaultdict
from math import log2
import bottleneck as bn
from experiments.src.sketches.basic.CountSketch import CountSketch
from experiments.src.sketches.abstract_sketch import Sketch

class CountSketchPlusPlus(Sketch): 
    n: int
    S: int
    preds: dict[np.int64,np.int64]
    small_tables: list[CountSketch]
    large_table: CountSketch


    def __init__(self, n: int, S: int, seed: int, preds: dict[np.int64,np.int64] = {}):
        np.random.seed(seed)
        self.n = n # num unique elements
        self.S = S # total space
        self.preds = preds
        L = 3
        S -= len(preds)
        if S <= 0:
            raise ValueError("Not enough space to store predictions. Either increase S or reduce number of predictions.")
        
        small_W = max(math.floor(S/ (log2(S) * 2*L)),3) 
        big_W = math.floor(S/(2*L))
        if small_W <3 or big_W < 3:
            raise ValueError("Not enough space for CountSketch tables.")

        self.small_tables = [CountSketch(n, small_W, L, preds=preds, seed=seed) for _ in range(math.floor(math.log2(S)))]
        self.large_table = CountSketch(n, big_W, L, preds=preds, seed=seed)

    def update(self, x: np.int64, d=1) -> None:
        """Update sketch with d copies of x""" 
        if x in self.preds:
            return 
        self.large_table.update(x, d)
        for table in self.small_tables:
            table.update(x, d)

    def vectorUpdate(self, freqs: np.ndarray) -> None:
        """Update sketch with a vector (frequency histogram)"""
        assert(len(freqs) == self.n)
        freqs[list(self.preds.keys())] = 0  # Ignore predictions in the update
        self.large_table.vectorUpdate(freqs)
        for table in self.small_tables:
            table.vectorUpdate(freqs)

    def get_small_and_big_table_estimates(self, x: np.int64) -> tuple[float, float]:
        """Estimate the frequency of x"""
        if x in self.preds: 
            raise ValueError(f"Element {x} is a prediction and cannot be estimated.")
        f_x_big_table = self.large_table.estimate(x)
        estimates = [table.estimate(x) for table in self.small_tables]
        f_x_small_tables = bn.median(np.array(estimates))
        return f_x_small_tables, f_x_big_table
    
    def estimate(self, x: np.int64, tau: int) -> float | np.int64:
        if not tau: 
            raise ValueError("Threshold tau must be provided for estimation.")
        if x in self.preds:
            return self.preds[x]
        f_x_small_table, f_x_big_table = self.get_small_and_big_table_estimates(x)
        if f_x_small_table < tau:
            return 0
        else:
            return f_x_big_table

    def evaluate(self, freq_vector: np.ndarray, abs_taus: np.ndarray) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
        """Evaluate the algorithm"""
        uerrs = defaultdict(lambda: np.zeros(len(freq_vector)))
        werrs = defaultdict(lambda: np.zeros(len(freq_vector)))
        for i, f_i in enumerate(freq_vector):
            if i in self.preds:
                for tau in abs_taus: 
                    uerrs[tau][i] = 0
                    werrs[tau][i] = 0
                continue
            f_x_small, f_x_big = self.get_small_and_big_table_estimates(np.int64(i))
            for tau in abs_taus:
                estimate = f_x_big if f_x_small >= tau else 0
                uerrs[tau][i] = abs(estimate - f_i)
                werrs[tau][i] = abs(estimate - f_i) * f_i

        return uerrs, werrs

    def run_and_evaluate_sketch_on_freq_vector(self, data: np.ndarray, abs_taus: np.ndarray):
        '''
        Evaluate a sketch on a given dataset which is a 
        vector of counts (the frequency histogram)
        '''
        self.vectorUpdate(data)
        uerrs_by_item, werrs_by_item = self.evaluate(data, abs_taus)
        uerr = np.array([bn.nanmean(uerrs_by_item[tau]) for tau in abs_taus])
        werr = np.array([bn.nanmean(werrs_by_item[tau]) for tau in abs_taus])
        return uerr, werr
    