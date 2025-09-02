""" Experiments Part 1: Comparing oracles in 2 pass setting
--------------------------------------

This experiments compares the quality of different oracles to predict the top k elements in a stream of data.
To do so, it runs and evaluates the CountSketch++ Sketch with predictions from different oracles. These oracles are: 
        - Space Savings Sketch (SpaceS)
        - Misra-Gries Sketch (MG)
        - Count Sketch (CS)
        - Count Min Sketch (CM)
        - CountSketch++ Sketch (CountSketch++)
"""

from collections import defaultdict
import logging
import bottleneck as bn
from experiments.src.utils import set_global_plot_style, get_markers
import os 
import yaml
import copy
import numpy as np
from experiments.config import settings
from experiments.src.sketches.basic.CountSketchPlusPlus import CountSketchPlusPlus
import matplotlib.pyplot as plt
from experiments.src.experiments import ExperimentSetUp, OracleCollection, Experiment
import copy

class ExperimentPart1(Experiment):
    def __init__(self): 
        self.uerrs_by_oracle_CountSketchPlusPlus = defaultdict(lambda: defaultdict(list))
        self.werrs_by_oracle_CountSketchPlusPlus = defaultdict(lambda: defaultdict(list))

        self.uerrs_by_oracle_0 = defaultdict(list)
        self.werrs_by_oracle_0 = defaultdict(list)

        self.werrs_CountSketchPlusPlus_wo_oracle = defaultdict(list)
        self.uerrs_CountSketchPlusPlus_wo_oracle = defaultdict(list)


    def run(self, setup: ExperimentSetUp,  oracles: OracleCollection | None = None):

        if setup.exp not in  ['e1', 'all']:
            raise ValueError(f"Experiment indicated in setup does not match! Setup: {setup.exp}, actual: e1.")

        logging.info("Running Experiment 1 with setup: %s", setup.get_params())

        if not oracles:
            oracles = OracleCollection.build_oracles(setup=setup)

        for S in setup.S_array:
            logging.info("Running for space S = %d", S)

            abs_taus = [(np.log2(S)/S)*c*setup.N for c in setup.threshConsts]

            CountSketchPlusPlus_Instance = CountSketchPlusPlus(setup.n, S, seed=setup.seed)
            CountSketchPlusPlus_Instance.vectorUpdate(setup.sorted_freq_vector)

            uerrs, werrs = CountSketchPlusPlus_Instance.evaluate(setup.sorted_freq_vector, np.array(abs_taus))
            for c, tau in zip(setup.threshConsts, abs_taus):
                c = float(c)
                self.uerrs_CountSketchPlusPlus_wo_oracle[c].append(float(bn.nansum(uerrs[tau])/setup.n))
                self.werrs_CountSketchPlusPlus_wo_oracle[c].append(float(bn.nansum(werrs[tau])/setup.n))

            
            for oracle_name, top_k_preds in oracles:
                freq_vector_copy = np.copy(setup.sorted_freq_vector)
                sketch_copy = copy.deepcopy(CountSketchPlusPlus_Instance)
                correcting_freq_vector = np.zeros(setup.n)
                correcting_freq_vector[list(top_k_preds[S].keys())] = -freq_vector_copy[list(top_k_preds[S].keys())]
                sketch_copy.vectorUpdate(correcting_freq_vector)

                uerrs, werrs = sketch_copy.evaluate(freq_vector_copy, np.array(abs_taus))
                for c, tau in zip(setup.threshConsts, abs_taus):
                    c = float(c)
                    self.uerrs_by_oracle_CountSketchPlusPlus[oracle_name][c].append(float(bn.nansum(np.delete(uerrs[tau], list(top_k_preds[S].keys())))/setup.n))
                    self.werrs_by_oracle_CountSketchPlusPlus[oracle_name][c].append(float(bn.nansum(np.delete(werrs[tau], list(top_k_preds[S].keys())))/setup.n))

                # Error when outputting zero for non-top-k elements
                rel_pred_errors = np.copy(setup.sorted_freq_vector)
                rel_pred_errors[list(top_k_preds[S].keys())] = 0
                self.uerrs_by_oracle_0[oracle_name].append(float(bn.nansum(rel_pred_errors)/setup.n))
                self.werrs_by_oracle_0[oracle_name].append(float(bn.nansum((rel_pred_errors*setup.sorted_freq_vector))/setup.n))

        output_path = os.path.join(setup.output_base_path, "e1", str(setup.start_time).split('.')[0], str(setup.seed))
        os.makedirs(output_path, exist_ok=True)
        self.write_logs(setup=setup, oracles=oracles, output_path=output_path)
        self.plot_single_run(setup=setup, output_path=output_path) 
        logging.info("\n ################## Experiments Part 1 completed successfully for seed %d. ✅ ################## \n", setup.seed)
        return output_path


    def write_logs(self, setup: ExperimentSetUp, oracles: OracleCollection, output_path: str):
        unweighted_errs = {k: dict(v) for k,v in self.uerrs_by_oracle_CountSketchPlusPlus.items()}
        for k in unweighted_errs.keys():
            unweighted_errs[k]['output 0'] = self.uerrs_by_oracle_0[k]

        weighted_errs = {k: dict(v) for k,v in self.werrs_by_oracle_CountSketchPlusPlus.items()}
        for k in weighted_errs.keys():
            weighted_errs[k]['output 0'] = self.werrs_by_oracle_0[k]

        weighted_errs["CountSketch++ wo Preds"] = dict(self.werrs_CountSketchPlusPlus_wo_oracle)
        unweighted_errs["CountSketch++ wo Preds"] = dict(self.uerrs_CountSketchPlusPlus_wo_oracle)

        errs = {
            'unweighted errors': unweighted_errs, 
            'weighted errors': weighted_errs
        }

        with open(output_path + '/oracles.yml', 'w') as outfile:
            yaml.dump(oracles.get_oracles(), outfile, default_flow_style=False, sort_keys=False)

        super().write_logs(setup=setup, errs=errs, output_path=output_path)
    


    def plot_single_run(self, setup: ExperimentSetUp, output_path: str):
        set_global_plot_style()

        self.plot_oracles_plus_CountSketchPlusPlus_by_tau(setup, output_path + "/oracles_plus_CountSketchPlusPlus_by_tau")
        self.plot_CountSketchPlusPlus_vs_0_by_oracle(setup, output_path + "/CountSketchPlusPlus_vs_0_by_oracle")


    def plot_CountSketchPlusPlus_vs_0_by_oracle(self, setup: ExperimentSetUp, output_path:str):

        os.makedirs(output_path, exist_ok=True)
        labels = [f"CountSketch++ (C={c})" for c in setup.threshConsts]
        oracle_names = list(self.uerrs_by_oracle_CountSketchPlusPlus.keys())

        
        fig, ax = plt.subplots(len(oracle_names),2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW*len(oracle_names)), constrained_layout=True)
        for ax_array in ax:
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(setup.S_array)))
                ax_i.set_xticklabels(setup.S_array)

        for oracle_name, ax_i in zip(oracle_names, ax[:,0]):
            ax_i.set_title('Unweighted Error; Oracle: {}'.format(oracle_name))

        for oracle_name, ax_i in zip(oracle_names, ax[:,1]):
            ax_i.set_title('Weighted Error; Oracle: {}'.format(oracle_name))

        # Plot lines
        markers = get_markers()
        for i, name in enumerate(oracle_names):
            ax[i,0].plot(self.uerrs_by_oracle_0[name], marker = markers[0], label="Output 0 for Non-Top-k")
            ax[i,1].plot(self.werrs_by_oracle_0[name], marker = markers[0], label="Output 0 for Non-Top-k")

            for j, c in enumerate(setup.threshConsts):
                ax[i,0].plot(self.uerrs_by_oracle_CountSketchPlusPlus[name][c], marker = markers[j+1], label=labels[j])
                ax[i,1].plot(self.werrs_by_oracle_CountSketchPlusPlus[name][c], marker = markers[j+1], label=labels[j])

        # Add global legend below
        handles, leg_labels = ax[0,0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            leg_labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),  # position to the right of the plot
            ncol=1,  # stacked vertically
            title="Frequency Estimation in 2nd Pass:"
        )
        legend.set_alignment("left")
        # Adjust layout to reserve space for legend
        plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.))
        plt.savefig(output_path + "/all_in_one.png", bbox_inches='tight')
        plt.close()

        for j, oracle_name in enumerate(oracle_names):
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
                
            ax.set_xlabel('Sketch Size')
            ax.set_ylabel('Weighted Error')
            ax.set_xticks(range(len(setup.S_array)))
            ax.set_xticklabels([str(s) for s in setup.S_array])

             # Plot lines
            markers = get_markers()

            ax.plot(self.werrs_by_oracle_0[oracle_name], marker = markers[0], label="Output 0 for Non-Top-k")

            for j, c in enumerate(setup.threshConsts):
                ax.plot(self.werrs_by_oracle_CountSketchPlusPlus[oracle_name][c], marker = markers[j+1], label=labels[j])

            # Add global legend below
            handles, leg_labels = ax.get_legend_handles_labels()
            legend = fig.legend(
                handles,
                leg_labels,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),  # position to the right of the plot
                ncol=1,  # stacked vertically
                title="Oracle: {}, Frequency Estimation in 2nd Pass:".format(oracle_name)
            )
            legend.set_alignment("left")
            # Adjust layout to reserve space for legend
            plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.))
            plt.savefig(output_path + f"/{oracle_name}.png", bbox_inches='tight')
            plt.close()

            

    def plot_oracles_plus_CountSketchPlusPlus_by_tau(self, setup: ExperimentSetUp, output_path: str):

        os.makedirs(output_path, exist_ok=True)
        labels = ["SS Oracle + CS++", "MG Oracle + CS++",  "CS Oracle + CS++", "CM Oracle + CS++", "CS++ Oracle + CS++", "Perfect Oracle + CS++", ]

        uerrs = list(self.uerrs_by_oracle_CountSketchPlusPlus.values())
        werrs = list(self.werrs_by_oracle_CountSketchPlusPlus.values())

        fig, ax = plt.subplots(len(setup.threshConsts),2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW*len(setup.threshConsts)), constrained_layout=True)
       
        for ax_array in ax:
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(setup.S_array)))
                ax_i.set_xticklabels(setup.S_array)

        for i, ax_i in enumerate(ax[:,0]):
            ax_i.set_title('Unweighted Error; C: {}'.format(setup.threshConsts[i]))

        for i,ax_i in enumerate(ax[:,1]):
            ax_i.set_title('Weighted Error; C: {}'.format(setup.threshConsts[i]))

        # Plot lines
        markers = get_markers()
        for j, c in enumerate(setup.threshConsts):
            ax[j,0].plot(self.uerrs_CountSketchPlusPlus_wo_oracle[c], marker = markers[0], label="CountSketch++ w/o Oracle")
            ax[j,1].plot(self.werrs_CountSketchPlusPlus_wo_oracle[c], marker = markers[0], label="CountSketch++ w/o Oracle" )

        for i, name in enumerate(labels):
            for j, c in enumerate(setup.threshConsts):
                ax[j,0].plot(uerrs[i][c], marker = markers[i+1], label=name)
                ax[j,1].plot(werrs[i][c], marker = markers[i+1], label=name)

        # Add global legend below
        handles, leg_labels = ax[0,0].get_legend_handles_labels()
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
                
            ax.set_xlabel('Sketch Size')
            ax.set_ylabel('Weighted Error')
            ax.set_xticks(range(len(setup.S_array)))
            ax.set_xticklabels([str(s) for s in setup.S_array])

             # Plot lines
            markers = get_markers()

            ax.plot(self.werrs_CountSketchPlusPlus_wo_oracle[c], marker = markers[0], label="CountSketch"
            "++ w/o Oracle")

            for i, name in enumerate(labels):
                ax.plot(werrs[i][c], marker = markers[i+1], label=name)

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
            plt.savefig(output_path + "/C_{}.png".format(c), bbox_inches='tight')
            plt.close()


                



