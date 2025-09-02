import math 
from experiments.config import settings
from typing import Literal
import numpy as np
from experiments.src.loader import load_data
from experiments.data import supported_source
from datetime import datetime 

class ExperimentSetUp(): 
     
    def __init__(self, 
                 start_time: datetime,
                 exp: Literal['e1', 'e2', 'e3', 'e4', 'all'],
                 data_src: supported_source,
                 threshConsts: np.ndarray, 
                 days: list[int] | None = None,  # Specify the days you want to include
                 n: int = 0,
                 seed: int = 42, 
                 ):
        
        if data_src == "AOL" and days is None:
            raise ValueError("For AOL data, 'days' must be specified. If you want to run the experiment on all days, hand over an empty list")
        
        if data_src == "Synthetic" and n == 0:
            raise ValueError("Number of distinct elements (n) not specified in method call. Must be a value larger than 0 if you choose synthetic data for the experiments.")
        
        sorted_freq_vector, stream = load_data(data_src, days=days, n=n, seed=seed)
        n = len(sorted_freq_vector) 
        B = math.ceil(math.log2(n))
        
        self.seed = seed
        self.start_time = start_time
        self.exp = exp
        self.data_src = data_src
        self.days = days  # Days for AOL data, None for synthetic or CAIDA
        self.B = B
        self.sorted_freq_vector = sorted_freq_vector
        self.stream = stream
        self.N = sum(sorted_freq_vector)  # Total number of elements in the stream
        self.n = n  # Number of unique elements in the stream
        self.numPreds = B
        self.S_array = [B*settings.SPACE_BASE*i for i in settings.SPACE_MULTIPLICATOR]
        #self.S_array = [B*i for i in range(5,7)]
        self.threshConsts = threshConsts
        self.output_base_path = settings.RESULTS_BASE_PATH  + data_src
        self.relative_heap_size_range = settings.RELATIVE_HEAP_SIZE_RANGE

    def get_params(self) -> dict:
        """
        Returns the parameters of the experiment as a dictionary.
        """
        params = {
            'n': self.n,
            'N': int(self.N),
            'S_array': self.S_array,
            'threshConsts': self.threshConsts.tolist(),
            'numPreds': self.numPreds,
            'seed': self.seed,
        }
        if self.data_src == "AOL":
            params['days'] = self.days if self.days is not None else "all" 
        return params
