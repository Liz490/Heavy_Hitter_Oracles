
from collections import defaultdict
import logging 
import numpy as np
from experiments.src.experiments import ExperimentSetUp, OracleCollection
import random 
import copy 
from datetime import datetime
import os 
from experiments.config import settings
from experiments.src.experiments.part_1.trials_exps_part_1 import MultiRunExperimentPart1
from experiments.src.experiments.exp_2_1.single_run_exp_2_1 import Exp_2_1
from experiments.src.experiments.exp_2_2.single_run_exp_2_2 import Exp_2_2

from experiments.src.experiments.exp_2_1.trials_exp_2_1 import MultiRunExperiment_2_1
from experiments.src.experiments.exp_2_2.trials_exp_2_2 import MultiRunExperiment_2_2
from experiments.src.experiments.overall_comparison.trials_overall_comp import MultiRunExpOverallComp



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    n = 50_000

    threshConsts = np.array([5, 8, 11])
    seeds = [0,42, 84, 168, 336]

    time = datetime.now()
    setup_by_seed = {seed: ExperimentSetUp(
        start_time=time,
        exp='all',
        data_src='Synthetic',
        threshConsts=threshConsts,
        n=n,
    ) for seed in seeds}
    
    for seed, setup in setup_by_seed.items():
        setup.seed = seed

    log_dirs = defaultdict(list)
    oracles_file_dir = os.path.join(settings.ORACLES_BASE_PATH, "Synthetic", "n_{}".format(n))
    #oracles = OracleCollection.build_oracles_for_seeds_and_save(setup_by_seed, file_dir=oracles_file_dir)
    
    oracles_by_seed = OracleCollection.load_oracles_by_seed(file_dir=oracles_file_dir, seeds=seeds)

    exp_part_1 = MultiRunExperimentPart1()
    log_dir_by_seed = exp_part_1.run(setup_by_seed,oracles_by_seed)
    for seed, log_dir in log_dir_by_seed.items():
        log_dirs[seed].append(log_dir)

    exp2_1_multi = MultiRunExperiment_2_1()
    log_dir_by_seed = exp2_1_multi.run(setup_by_seed, oracles_by_seed)
    for seed, log_dir in log_dir_by_seed.items():
        log_dirs[seed].append(log_dir)

    exp3_multi = MultiRunExperiment_2_2()
    log_dir_by_seed = exp3_multi.run(setup_by_seed, oracles_by_seed)
    for seed, log_dir in log_dir_by_seed.items():
        log_dirs[seed].append(log_dir)

    exp_overall_comp_multi = MultiRunExpOverallComp()
    exp_overall_comp_multi.run(setup_by_seed, log_dirs)
