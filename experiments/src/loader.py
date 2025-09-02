import pandas as pd 
from experiments.config import settings
import numpy as np
import math
from experiments.data import supported_source
from experiments.src.preprocessing.aol import compute_aol_ground_truth
from experiments.src.preprocessing.caida import compute_caida_ground_truth

def _generate_zipfian_stream(freq_vector: np.ndarray) -> np.ndarray:
    stream = np.repeat(np.arange(len(freq_vector)), freq_vector)
    np.random.shuffle(stream)
    return stream
    

def generateZipf(n, a=1) -> tuple[np.ndarray, np.ndarray]:
    '''
    Create data corresponding to a Zipf distribution
    over n elements with freq n/i^a for element i
    
    Calculate the error of the sketch
    '''
    sorted_freq_vector =  np.array([math.ceil(n/((i+1)**a)) for i in range(0, n)])
    stream = _generate_zipfian_stream(sorted_freq_vector)
    return sorted_freq_vector, stream


def loadAOL(days: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    # days starts at 0, i.e. 0 is the first day of recorded data 
    data = pd.read_csv(settings.AOL_CLEANED_DATA)
    ground_truth = pd.read_csv(settings.AOL_GROUND_TRUTH) 
    if days: 
        # Filter the data for the specified days
        data = data[data['AbsDay'].isin(days)]
        ground_truth = compute_aol_ground_truth(data)  
    sorted_freq_vector = ground_truth['Count'].to_numpy()
    stream = ground_truth["Rank"].to_numpy()
    np.random.shuffle(stream) 

    return sorted_freq_vector, stream


def loadCAIDA() -> tuple[np.ndarray, np.ndarray]:
    ground_truth = pd.read_csv(settings.CAIDA_GROUND_TRUTH)
    data = pd.read_csv(settings.CAIDA_CLEANED_DATA)
    freq_vector = ground_truth['Count'].to_numpy()
    stream = data["Rank"].to_numpy()
    np.random.shuffle(stream)

    return freq_vector, stream


def aol_iterator():
    data_array = load_el_stream_by_rank()
    for entry in data_array:
        yield entry

def load_el_stream_by_rank() -> np.ndarray: 
    el_stream = pd.read_csv(settings.AOL_CLEANED_DATA)["Rank"]
    np_stream = el_stream.to_numpy()
    return np_stream


def load_data(source: supported_source, seed:int, days: list[int] | None, n:int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from the specified source.
    """
    np.random.seed(seed)
    if source == "AOL":
        return loadAOL(days=days)
    elif source == "CAIDA":
        return loadCAIDA()
    elif source == "Synthetic":
        if n == 0: 
            raise ValueError("Number of distinct elements (n) not specified in method call. Must be a value larger 0 if you choose synthetic data for the experiments.")
        return generateZipf(n)  # Example size, can be adjusted
    else:
        raise ValueError(f"Unsupported data source: {source}")

