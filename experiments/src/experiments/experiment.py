"""Abstract base class for experiments."""

from abc import ABC, abstractmethod
from datetime import datetime 
import yaml
import logging
from experiments.src.experiments import ExperimentSetUp

class Experiment(ABC):
    """
    Abstract base class for experiments.
    All experiments should inherit from this class and implement the run method.
    """

    @abstractmethod
    def run(self, setup, *args, **kwargs):
        """
        Run the experiment with the given setup.
        
        Args:
        ----------
            setup (ExperimentSetUp): The setup for the experiment.
        """
        pass


    @abstractmethod
    def plot_single_run(self, setup: ExperimentSetUp):
        """
        Plot the results of the experiment.
        Args:
        ----------
            setup (ExperimentSetUp): The setup for the experiment.
        """
        pass


    def write_logs(self, setup: ExperimentSetUp, errs: dict, output_path: str):
        with open(output_path + '/params.yml', 'w') as outfile:
                params = setup.get_params()
                params['execution time'] = str((datetime.now() - setup.start_time)).split(".")[0]  + "(HH:MM:SS)" # Format as HH:MM:SS
                yaml.dump(params, outfile,default_flow_style=False, sort_keys=False)

        with open(output_path + '/errors.yml', 'w') as outfile:
            yaml.dump(errs, outfile,default_flow_style=False, sort_keys=False)

        logging.info("Results and params logged to %s", setup.output_base_path)
