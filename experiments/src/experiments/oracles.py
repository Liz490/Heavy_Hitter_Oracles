import logging 
import numpy as np
import yaml
from datetime import datetime
import os
from experiments.config import settings
import random
from experiments.src.sketches import CountSketchHeap, SpaceSavings, MisraGries, CountSketchPlusPlusHeap, CountMinHeap
from experiments.src.experiments import ExperimentSetUp


class OracleCollection: 

    def __init__(self):
        self.top_k_SpaceS: dict[float,dict[int, int]] = dict()
        self.top_k_MG: dict[float,dict[int, int]] = dict()
        self.top_k_CS: dict[float,dict[int, int]] = dict()
        self.top_k_CM: dict[float,dict[int, int]] = dict()
        self.top_k_MIT: dict[float,dict[int, int]] = dict()
        self.top_k_perfect: dict[float,dict[int, int]] = dict()

    def __iter__(self):
        oracle_names = ["SpaceSavings", "MisraGries", "CountSketch", "CountMin", "MIT", "perfect"]
        top_k_values = [self.top_k_SpaceS, self.top_k_MG, self.top_k_CS, self.top_k_CM, self.top_k_MIT, self.top_k_perfect]
        yield from zip(oracle_names, top_k_values)

    @classmethod
    def load_from_file(cls, file_dir: str, seed: int) -> 'OracleCollection':
        """
        Load the oracles from a file.
        
        Parameters:
        -----------
            file_path: str
                The path to the file containing the oracles.
        """
        file_path = os.path.join(file_dir, f'seed_{seed}.yaml')
        oracle_collection = cls()
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            oracle_collection.top_k_SpaceS = data['SpaceSavings']
            oracle_collection.top_k_MG = data['MisraGries']
            oracle_collection.top_k_CS = data['CountSketch']
            oracle_collection.top_k_CM = data['CountMin']
            oracle_collection.top_k_MIT = data['MIT']
            oracle_collection.top_k_perfect = data['perfect']
        return oracle_collection    
    
    def get_oracles(self):
        """
        Returns the oracles as a dictionary.
        """
        return {
            "SpaceSavings": {k: dict(v) for k,v in self.top_k_SpaceS.items()},
            "MisraGries": {k: dict(v) for k,v in self.top_k_MG.items()},
            "CountSketch": {k: dict(v) for k,v in self.top_k_CS.items()},
            "CountMin": {k: dict(v) for k,v in self.top_k_CM.items()},
            "MIT": {k: dict(v) for k,v in self.top_k_MIT.items()},
            "perfect": {k: dict(v) for k,v in self.top_k_perfect.items()}

        }
    
    def save_to_file(self, file_dir: str, seed: int):
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, f'seed_{seed}.yaml')
        with open(file_path, 'w') as file:
            yaml.dump(self.get_oracles(), file, default_flow_style=False, sort_keys=False)
        logging.info("Oracles saved to %s", file_path)

    @classmethod
    def build_oracles(cls, setup: ExperimentSetUp): 
        """
        Run the oracles for the sketches and store the results.

        Returns:
        --------
            OracleCollection: An instance of OracleCollection containing the top k elements for each sketch and space size.
        """

        logging.info("Running oracles for sketches with k = %d, seed = %d", setup.numPreds, setup.seed)

        oracle_collection = cls()

        np.random.seed(setup.seed)
        random.seed(setup.seed) 

        for S in setup.S_array:
            logging.info("Running for space S = %d", S)

            spaceS = SpaceSavings(S= S)
            misraGries = MisraGries(S=S)
            MIT_heap = CountSketchPlusPlusHeap(n=setup.n, S_total=S , seed = setup.seed, S_heap=setup.numPreds)
            CS_heap = CountSketchHeap(n=setup.n, S_total=S, seed=setup.seed, S_heap=setup.numPreds)
            CM_heap = CountMinHeap(n=setup.n, S_total=S, seed=setup.seed, S_heap=setup.numPreds)

            # Build the sketches/ oracle 
            for x in setup.stream:
                spaceS.update(x)
                misraGries.update(x)
                MIT_heap.update(x)
                CS_heap.update(x)
                CM_heap.update(x)
            
            # Retrieve top k elements
            oracle_collection.top_k_SpaceS[S] = spaceS.get_top_k_with_freqs(setup.numPreds)
            oracle_collection.top_k_MIT[S] = MIT_heap.get_top_k_with_freqs(setup.numPreds)
            oracle_collection.top_k_CS[S] = CS_heap.get_top_k_with_freqs(setup.numPreds)
            oracle_collection.top_k_MG[S] = misraGries.get_top_k_with_freqs(setup.numPreds)
            oracle_collection.top_k_CM[S] = CM_heap.get_top_k_with_freqs(setup.numPreds)
            oracle_collection.top_k_perfect[S] = {i: int(setup.sorted_freq_vector[i]) for i in range(setup.numPreds)}

        logging.info("Oracles built. (Seed: %d) ✅", setup.seed)
        return oracle_collection


    @classmethod
    def load_oracles_by_seed(cls, file_dir: str, seeds: list[int]) -> dict[int, 'OracleCollection']:
        """
        Load oracles for multiple seeds.

        Returns:
        --------
            dict[int, OracleCollection]: A dictionary mapping each seed to its corresponding OracleCollection.
        """
        oracle_collections_by_seed = {}
        for seed in seeds:
            oracle_collections_by_seed[seed] = cls.load_from_file(file_dir=file_dir, seed=seed)
            logging.info("Oracles loaded for seed %d", seed)

        return oracle_collections_by_seed

    
    @classmethod
    def build_oracles_for_seeds_and_save(cls, setup_by_seed: dict[int, ExperimentSetUp], file_dir: str) -> dict[int, 'OracleCollection']:
        """
        Build oracles for multiple seeds.

        Returns:
        --------
            dict[int, OracleCollection]: A dictionary mapping each seed to its corresponding OracleCollection.
        """
        start_time = datetime.now()
        oracle_collections = {}
        for seed, setup in setup_by_seed.items():
            np.random.seed(seed)
            start_time_single_seed = datetime.now()
            current_collection = cls.build_oracles(setup=setup)
            oracle_collections[seed] = current_collection
            current_collection.save_to_file(file_dir=file_dir, seed=seed)
            end_single_seed = datetime.now()
            duration = round((end_single_seed - start_time_single_seed).total_seconds() / 60, 2)
            logging.info("Oracles for seed %d built and saved in %s minutes", seed, duration)
        end_time = datetime.now()
        total_duration = round((end_time - start_time).total_seconds() / 60, 2)
        logging.info("All oracles built and saved in %s minutes", total_duration)
        return oracle_collections
