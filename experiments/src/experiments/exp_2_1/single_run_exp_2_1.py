""" Experiment 2.1: Using Oracle Counts as Frequency Estimates
---------------------------------------
This experiments runs oracles and uses the associated counts as frequency estimates. 
For all elements that are not in the predicted top-k, the algorithm returns 0 as frequency estimate.
The experiment compares the following oracles:
        - Space Savings Sketch (SpaceS)
        - Misra-Gries Sketch (MG)
        - Count Sketch (CS)
        - Count Min Sketch (CM)
        - CountSketch++ Sketch (CS++)
"""

from collections import defaultdict
from datetime import datetime
import logging
import os 
from experiments.config import  settings
import bottleneck as bn
from matplotlib import cm
import yaml
import copy
from experiments.src.utils import set_global_plot_style, get_markers
import numpy as np
from experiments.src.sketches.basic.CountSketchPlusPlus import CountSketchPlusPlus
import matplotlib.pyplot as plt
from experiments.src.experiments import ExperimentSetUp, OracleCollection, Experiment



class Exp_2_1(Experiment):
    def __init__(self): 
        self.uerrs_by_oracle = defaultdict(list)
        self.werrs_by_oracle = defaultdict(list)

        self.uerrs_CountSketchPlusPlus_by_tau = defaultdict(list)
        self.werrs_CountSketchPlusPlus_by_tau = defaultdict(list)

    def run(self, setup: ExperimentSetUp, oracles: OracleCollection| None = None):

        if setup.exp not in  ['e2', 'all']:
            raise ValueError(f"Experiment indicated in setup does not match! Setup: {setup.exp}, actual: e2.")

        logging.info("Running Experiment 2.1 with setup: %s", setup.get_params())

        if not oracles:
            oracles = OracleCollection.build_oracles(setup=setup)

        for S in setup.S_array:
            logging.info("Running for space S = %d", S)

            abs_taus = np.array([(np.log2(S)/S)*c*setup.N for c in setup.threshConsts])

            CountSketchPlusPlus_instance = CountSketchPlusPlus(setup.n, S, setup.seed)
            uerrs, werrs = CountSketchPlusPlus_instance.run_and_evaluate_sketch_on_freq_vector(setup.sorted_freq_vector, abs_taus)
            for i, c in enumerate(setup.threshConsts):
                c= float(c)
                self.uerrs_CountSketchPlusPlus_by_tau[c].append(float(uerrs[i]))
                self.werrs_CountSketchPlusPlus_by_tau[c].append(float(werrs[i]))

            for oracle_name, top_k_preds in oracles:
                pred_freq_vector = np.copy(setup.sorted_freq_vector)
                pred_freq_vector[list(top_k_preds[S].keys())] = abs(pred_freq_vector[list(top_k_preds[S].keys())] - list(top_k_preds[S].values()))  # Set frequencies of non-top-k elements to their predicted values

                self.uerrs_by_oracle[oracle_name].append(float(bn.nanmean(pred_freq_vector)))
                self.werrs_by_oracle[oracle_name].append(float(bn.nanmean(pred_freq_vector*setup.sorted_freq_vector)))

        output_path = os.path.join(setup.output_base_path, "e2", str(setup.start_time).split('.')[0], str(setup.seed) )
        output_path_plots = os.path.join(output_path, "plots")
        os.makedirs(output_path_plots, exist_ok=True)
        self.write_logs(setup=setup, oracles=oracles, output_path=output_path)
        self.plot_single_run(setup=setup, output_path=output_path_plots) 
        logging.info("\n ################## Experiment 2.1 completed successfully for seed %d. ✅ ################## \n", setup.seed)
        return output_path

    
    def write_logs(self,setup: ExperimentSetUp, oracles: OracleCollection, output_path: str):
        errs = {
            'unweighted errors (oracle estimates + outputting 0)': dict(self.uerrs_by_oracle),
            'weighted errors (oracle estimates + outputting 0)': dict(self.werrs_by_oracle),
            'unweighted errors by (CountSketch++ without predictions)': dict(self.uerrs_CountSketchPlusPlus_by_tau),
            'weighted errors by (CountSketch++ without predictions)': dict(self.werrs_CountSketchPlusPlus_by_tau)
        }

        with open(output_path + '/oracles.yml', 'w') as outfile:
            yaml.dump(oracles.get_oracles(), outfile, default_flow_style=False, sort_keys=False)

        super().write_logs(setup=setup, errs=errs, output_path=output_path)


    def plot_single_run(self, setup: ExperimentSetUp, output_path: str):
        set_global_plot_style()


        labels = ["Space Savings Oracle", "Misra Gries Oracle", "Count Sketch Oracle", "Count Min Oracle", "CountSketch++ Oracle"]

        uerrs = list(self.uerrs_by_oracle.values())
        werrs = list(self.werrs_by_oracle.values())

        fig, ax = plt.subplots(len(setup.threshConsts), 2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW * len(setup.threshConsts)), constrained_layout=True)

        for i, ax_array in enumerate(ax):
            ax_array[0].set_title('Unweighted Error (C={})'.format(setup.threshConsts[i]))
            ax_array[1].set_title('Weighted Error (C={})'.format(setup.threshConsts[i]))
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(setup.S_array)))
                ax_i.set_xticklabels(setup.S_array)

        # Plot lines
        markers = get_markers()
        
        for j, c in enumerate(setup.threshConsts):
            ax[j,0].plot(self.uerrs_CountSketchPlusPlus_by_tau[c], marker = markers[0], label=f"CountSketch++ w/o Preds", )
            ax[j,1].plot(self.werrs_CountSketchPlusPlus_by_tau[c], marker = markers[0], label=f"CountSketch++ w/o Preds", )
            for i, name in enumerate(labels):
                ax[j,0].plot(uerrs[i], marker = markers[i+1], label=name)
                ax[j,1].plot(werrs[i], marker = markers[i+1], label=name)
            ax[j,0].axhline(y=self.uerrs_by_oracle["perfect"][0], linestyle='--', label='Perfect Oracle', color= cm.get_cmap('Paired')(8))
            ax[j,1].axhline(y=self.werrs_by_oracle["perfect"][0], linestyle='--', label='Perfect Oracle', color= cm.get_cmap('Paired')(8))
           
               

       # Add global legend below
        handles, leg_labels = ax[0][0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            leg_labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),  # position to the right of the plot
            ncol=1  # stacked vertically
        )
        legend.set_alignment("left")
        # Adjust layout to reserve space for legend
        plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.))
        plt.savefig(output_path + "/all_in_one.png", bbox_inches='tight')
        plt.close()

        for j, c in enumerate(setup.threshConsts):
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

            ax.set_ylabel('Weighted Error')
            ax.set_xlabel('Sketch Size')
            ax.set_xticks(range(len(setup.S_array)))
            ax.set_xticklabels([str(s) for s in setup.S_array])


            # Plot lines
            markers = get_markers()
            ax.plot(self.werrs_CountSketchPlusPlus_by_tau[c], marker = markers[0], label=f"CountSketch++ w/o Preds", )
            for i, name in enumerate(labels):
                ax.plot(werrs[i], marker = markers[i+1], label=name)
            ax.axhline(y=self.werrs_by_oracle["perfect"][0], linestyle='--', label='Perfect Oracle', color= cm.get_cmap('Paired')(8))
            

            # Add global legend below
            handles, leg_labels = ax.get_legend_handles_labels()
            legend = fig.legend(
                handles,
                leg_labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),  # position to the right of the plot
                ncol=1  # stacked vertically
            )
            legend.set_alignment("left")
            # Adjust layout to reserve space for legend
            plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.))
            plt.savefig(output_path + f"/w_err_only_c_{c}.png", bbox_inches='tight')
            plt.close()
        
