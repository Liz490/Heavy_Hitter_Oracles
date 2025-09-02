""" Experiment 2.2: Comparing Augmented Oracles in 1 Pass Setting with CountSketch++ without Predictions
--------------------------------------

This experiments runs oracles alongside a separate CountSketch++ Sketch. When estimating the frequency for element x, 
it returns the estimate from the oracle in case x is in the top-k predictions of the oracle.
Otherwise it returns the estimate from the CountSketch++ Sketch.
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
import yaml
import copy
import numpy as np
import matplotlib.cm as cm
from experiments.src.utils import set_global_plot_style, get_markers
from experiments.src.sketches.basic.CountSketchPlusPlus import CountSketchPlusPlus
import matplotlib.pyplot as plt
from experiments.src.experiments import ExperimentSetUp, OracleCollection, Experiment



class Exp_2_2(Experiment):
    def __init__(self): 
        self.uerrs_by_oracle = defaultdict(lambda: defaultdict(list))
        self.werrs_by_oracle = defaultdict(lambda: defaultdict(list))

        self.uerrs_CountSketchPlusPlus_wo_preds_by_tau = defaultdict(list)
        self.werrs_CountSketchPlusPlus_wo_preds_by_tau = defaultdict(list)

    def run(self, setup: ExperimentSetUp, oracles: OracleCollection| None = None):

        if setup.exp not in  ['e3', 'all']:
            raise ValueError(f"Experiment indicated in setup does not match! Setup: {setup.exp}, actual: e3.")

        logging.info("Running Experiment 3 with setup: %s", setup.get_params())

        if not oracles:
            oracles = OracleCollection.build_oracles(setup=setup)

        for S in setup.S_array:
            logging.info("Running for space S = %d", S)

            abs_taus = np.array([(np.log2(S)/S)*c*setup.N for c in setup.threshConsts])

            CountSketchPlusPlus_sketch = CountSketchPlusPlus(setup.n, S, seed = setup.seed)
            CountSketchPlusPlus_sketch.vectorUpdate(setup.sorted_freq_vector)  
            uerrs_by_item, werrs_by_item = CountSketchPlusPlus_sketch.evaluate(setup.sorted_freq_vector, abs_taus)
            uerr_CountSketchPlusPlus_wo_preds_by_tau = np.array([bn.nanmean(uerrs_by_item[tau]) for tau in abs_taus])
            werr_CountSketchPlusPlus_wo_preds_by_tau = np.array([bn.nanmean(werrs_by_item[tau]) for tau in abs_taus])

            for i, c in enumerate(setup.threshConsts):
                c= float(c)
                self.uerrs_CountSketchPlusPlus_wo_preds_by_tau[c].append(float(uerr_CountSketchPlusPlus_wo_preds_by_tau[i]))
                self.werrs_CountSketchPlusPlus_wo_preds_by_tau[c].append(float(werr_CountSketchPlusPlus_wo_preds_by_tau[i]))


                for oracle_name, top_k_preds in oracles:
                    freq_vector_top_k = np.copy(setup.sorted_freq_vector[list(top_k_preds[S].keys())])
                    sum_uerr_top_k = sum(abs(freq_vector_top_k - list(top_k_preds[S].values())))
                    sum_werr_top_k = sum(abs((freq_vector_top_k - list(top_k_preds[S].values()))) * freq_vector_top_k)
                    u_sketch_estimates = np.array(uerrs_by_item[abs_taus[i]])
                    w_sketch_estimates = np.array(werrs_by_item[abs_taus[i]])
                    sum_uerr_non_top_k = sum(np.delete(u_sketch_estimates, list(top_k_preds[S].keys())))
                    sum_werr_non_top_k = sum(np.delete(w_sketch_estimates, list(top_k_preds[S].keys())))
                    uerr_total = (sum_uerr_top_k + sum_uerr_non_top_k) / setup.n
                    werr_total = (sum_werr_top_k + sum_werr_non_top_k) / setup.n

                    self.uerrs_by_oracle[oracle_name][c].append(float(uerr_total))
                    self.werrs_by_oracle[oracle_name][c].append(float(werr_total))


        output_path = os.path.join(setup.output_base_path, "e3", str(setup.start_time).split('.')[0], str(setup.seed))
        output_path_plot = os.path.join(output_path, "plots")
        os.makedirs(output_path_plot, exist_ok=True)
        self.write_logs(setup=setup, oracles=oracles, output_path=output_path)
        self.plot_single_run(setup=setup, output_path=output_path_plot)
        logging.info("\n ################## Experiment 2.2 completed successfully for seed %d. ✅ ################## \n", setup.seed)
        return output_path

    
    def write_logs(self, setup: ExperimentSetUp, oracles: OracleCollection, output_path: str):
        errs = {
            'unweighted errors (oracle + sketch)': {k: dict(v) for k, v in self.uerrs_by_oracle.items()},
            'weighted errors (oracle + sketch)': {k: dict(v) for k, v in self.werrs_by_oracle.items()},
            'unweighted errors by (CountSketch++ without predictions)': dict(self.uerrs_CountSketchPlusPlus_wo_preds_by_tau),
            'weighted errors by (CountSketch++ without predictions)': dict(self.werrs_CountSketchPlusPlus_wo_preds_by_tau)
        }

        with open(output_path + '/oracles.yml', 'w') as outfile:
            yaml.dump(oracles.get_oracles(), outfile, default_flow_style=False, sort_keys=False)

        super().write_logs(setup=setup, errs=errs, output_path=output_path)


    def plot_single_run(self, setup: ExperimentSetUp, output_path: str):
        set_global_plot_style()

        labels = ["Space Savings Oracle (augm.)", "Misra Gries Oracle (augm.)", "Count Sketch Oracle (augm.)", "Count Min Oracle (augm.)", "CountSketch++ Oracle (augm.)"]

        uerrs = list(self.uerrs_by_oracle.values()) 
        werrs = list(self.werrs_by_oracle.values())

        fig, ax = plt.subplots(len(setup.threshConsts),2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW*len(setup.threshConsts)), constrained_layout=True)

        for i, ax_array in enumerate(ax):
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            ax_array[0].set_title('Unweighted Error (C={})'.format(setup.threshConsts[i]))
            ax_array[1].set_title('Weighted Error (C={})'.format(setup.threshConsts[i]))
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(setup.S_array)))
                ax_i.set_xticklabels(setup.S_array)

        # Plot lines
        markers = get_markers()
        for j, c in enumerate(setup.threshConsts):
            ax[j,0].plot(self.uerrs_CountSketchPlusPlus_wo_preds_by_tau[c], marker = markers[0], label=f"CountSketch++ w/0 Preds")
            ax[j,1].plot(self.werrs_CountSketchPlusPlus_wo_preds_by_tau[c], marker = markers[0], label=f"CountSketch++ w/0 Preds")

            for i, name in enumerate(labels):
                ax[j,0].plot(uerrs[i][c], marker = markers[i+1], label=name)
                ax[j,1].plot(werrs[i][c], marker = markers[i+1], label=name)

            ax[j,0].axhline(y=self.uerrs_by_oracle["perfect"][c][0], linestyle='--', label='Perfect Oracle', color= cm.get_cmap('Paired')(8))
            ax[j,1].axhline(y=self.werrs_by_oracle["perfect"][c][0], linestyle='--', label='Perfect Oracle', color= cm.get_cmap('Paired')(8))


            
        # Add global legend below
        handles, leg_labels = ax[0,0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            leg_labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),  # position to the right of the plot
            ncol=1,  # stacked vertically
            title="Augmented Oracles (1 Pass):"
        )
        legend.set_alignment("left")
        # Adjust layout to reserve space for legend
        plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.))
        plt.savefig(output_path + "/all_in_one.png", bbox_inches='tight')
        plt.close()


        for i, c in enumerate(setup.threshConsts):
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
                
            ax.set_xlabel('Sketch Size')
            ax.set_ylabel('Weighted Error')
            ax.set_xticks(range(len(setup.S_array)))
            ax.set_xticklabels([str(s) for s in setup.S_array])

             # Plot lines
            markers = get_markers()
            ax.plot(self.uerrs_CountSketchPlusPlus_wo_preds_by_tau[c], marker = markers[0], label="CountSketch++ w/o Oracle")

            for i, name in enumerate(labels):
                ax.plot(werrs[i][c], marker = markers[i+1], label=name)
            ax.axhline(y=self.uerrs_by_oracle["perfect"][c][0], linestyle='--', label='Perfect Oracle', color= cm.get_cmap('Paired')(8))
           


            # Add global legend below
            handles, leg_labels = ax.get_legend_handles_labels()
            legend = fig.legend(
                handles,
                leg_labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),  # position to the right of the plot
                ncol=1,  # stacked vertically
                title="Augmented Oracles (1 Pass):"
            )
            legend.set_alignment("left")
            # Adjust layout to reserve space for legend
            plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.))
            plt.savefig(output_path + "/C_{}.png".format(c), bbox_inches='tight')
            plt.close()
