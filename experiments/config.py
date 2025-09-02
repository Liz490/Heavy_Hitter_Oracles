from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pathlib import Path

class Config(BaseSettings): 
    FIG_HEIGHT_PER_ROW: int = 4
    FIG_WIDTH_PER_COL: int = 6
    
    SYNTHETIC_DISTR_PLOT_PATH: str

    AOL_RAW_DATA: Path
    AOL_CLEANED_DATA: Path
    AOL_GROUND_TRUTH: Path
    AOL_RESULTS_BASE_PATH: Path
    AOL_DISTR_PLOT_PATH: Path
    AOL_QUERY_DISTR_PLOT_PATH: Path

    COMBINED_DISTRIBUTIONS_PLOT_PATH: Path

    CAIDA_RAW_DATA: str
    CAIDA_DATA_COMPRESSED: str
    CAIDA_GROUND_TRUTH: str
    CAIDA_RESULTS_BASE_PATH: str
    CAIDA_CLEANED_DATA: str
    CAIDA_QUERY_DISTR_PLOT_PATH: Path


    RESULTS_BASE_PATH: str 

    ORACLES_BASE_PATH: str

    PRIME: int = 2**20 - 1
    N_AOL: int  = 1_244_494
    AOL_STREAM_LENGTH: int = 2_783_771
    MAX_PARALLEL_WORKERS: int = 32
    EL_STREAM_BY_RANK_PATH: str = "data/aol_el_by_rank_stream.npy"

    SPACE_MULTIPLICATOR: list[int] = [1,2,5,10,20]
    SPACE_BASE: int = 7

    RELATIVE_HEAP_SIZE_RANGE: list[float] = [0.1, 0.2, 0.5, 0.6]


def get_config() -> Config:
    load_dotenv(override=True)
    return Config()

settings = get_config()
