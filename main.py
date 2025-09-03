
import logging 
from experiments.src.reproduction_scripts.caida import reproduce_experiments_on_caida_data
from experiments.src.reproduction_scripts.aol import reproduce_experiments_on_aol_data
from experiments.src.reproduction_scripts.synthetic import reproduce_experiments_on_synthetic_data


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    reproduce_experiments_on_synthetic_data()
    #reproduce_experiments_on_aol_data()
    #reproduce_experiments_on_caida_data()
    