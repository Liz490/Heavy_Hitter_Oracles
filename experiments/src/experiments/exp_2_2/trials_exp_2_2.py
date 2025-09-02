from experiments.src.experiments.setup import ExperimentSetUp
from experiments.src.experiments.oracles import OracleCollection
from experiments.src.experiments.exp_2_2.single_run_exp_2_2 import Exp_2_2
from collections import defaultdict
import logging
import numpy as np
import matplotlib.pyplot as plt
from experiments.src.utils import get_markers, set_global_plot_style
import os
from experiments.config import  settings
import matplotlib.cm as cm
from experiments.src.utils import set_global_plot_style

class MultiRunExperiment_2_2:
    def __init__(self):
        self.trials_uerrs_by_oracle = defaultdict(lambda: defaultdict(list))
        self.trials_werrs_by_oracle = defaultdict(lambda: defaultdict(list))

        self.trials_uerrs_CountSketchPlusPlus_wo_preds_by_tau = defaultdict(list)
        self.trials_werrs_CountSketchPlusPlus_wo_preds_by_tau = defaultdict(list)


    def update(self, trial_run: Exp_2_2):
        for oracle_name, v in trial_run.uerrs_by_oracle.items():
            for thres_const in v.keys():
                self.trials_uerrs_by_oracle[oracle_name][thres_const].append(v[thres_const])

        for oracle_name, v in trial_run.werrs_by_oracle.items():
            for thres_const in v.keys():
                self.trials_werrs_by_oracle[oracle_name][thres_const].append(v[thres_const])

        for thres_const, v in trial_run.uerrs_CountSketchPlusPlus_wo_preds_by_tau.items():
            self.trials_uerrs_CountSketchPlusPlus_wo_preds_by_tau[thres_const].append(v)

        for thres_const, v in trial_run.werrs_CountSketchPlusPlus_wo_preds_by_tau.items():
            self.trials_werrs_CountSketchPlusPlus_wo_preds_by_tau[thres_const].append(v)


    def run(self, setup_by_seed: dict[int, ExperimentSetUp], oracles_by_seed: dict[int, OracleCollection] | None = None) -> dict[int, str]:
        """
        Run the experiment with the given setup and oracles for multiple seeds.
        
        Args:
        ----------
            setup (ExperimentSetUp): The setup for the experiment.
            seeds (list[int]): List of seeds for different trials.
            oracles (dict[int, OracleCollection] | None): Dictionary of oracles for each seed. If None, oracles will be built.

        """
        output_paths_by_seed = {}
        for seed, setup in setup_by_seed.items():
            np.random.seed(seed)
            logging.info(f"\n\nStarting trial with seed {seed}...\n")
            
            oracles = oracles_by_seed[seed] if oracles_by_seed else None

            # Reset per-trial instance (necessary to avoid carry-over)
            trial_run = Exp_2_2()
            output_path = trial_run.run(setup=setup, oracles=oracles)
            output_paths_by_seed[seed] = output_path

            self.update(trial_run)
        
        threshConsts = setup_by_seed[next(iter(setup_by_seed))].threshConsts
        S_array = setup_by_seed[next(iter(setup_by_seed))].S_array
        output_base_path = setup_by_seed[next(iter(setup_by_seed))].output_base_path
        start_time = setup_by_seed[next(iter(setup_by_seed))].start_time

        output_dir = os.path.join(output_base_path, "e3",  str(start_time).split('.')[0])
        self.aggregate_and_plot_trials_part_3(threshConsts, S_array, output_dir)

        logging.info("\n \n %d trials completed for experiment 2.2. ✅ \n", len(setup_by_seed))
        return output_paths_by_seed

    def aggregate_and_plot_trials_part_3(self, threshConsts, S_array, output_dir: str):
        set_global_plot_style()

        def mean_std_across_trials(list_of_lists):
            arr = np.array(list_of_lists)
            return np.mean(arr, axis=0), np.std(arr, axis=0) 
        

        os.makedirs(output_dir, exist_ok=True)

        labels = ["Space Savings Oracle (augm.)", "Misra Gries Oracle (augm.)", "Count Sketch Oracle (augm.)", "Count Min Oracle (augm.)", "CountSketch++ Oracle (augm.)"]

        uerrs = list(self.trials_uerrs_by_oracle.values()) 
        werrs = list(self.trials_werrs_by_oracle.values())

        fig, ax = plt.subplots(len(threshConsts),2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW*len(threshConsts)), constrained_layout=True)

        for i, ax_array in enumerate(ax):
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            ax_array[0].set_title('Unweighted Error (C={})'.format(threshConsts[i]))
            ax_array[1].set_title('Weighted Error (C={})'.format(threshConsts[i]))
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(S_array)))
                ax_i.set_xticklabels(S_array)

        # Plot lines
        markers = get_markers()
        for j, c in enumerate(threshConsts):

            u_vals_wo, u_std_wo = mean_std_across_trials(self.trials_uerrs_CountSketchPlusPlus_wo_preds_by_tau[c])
            w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werrs_CountSketchPlusPlus_wo_preds_by_tau[c])
            ax[j][0].plot(u_vals_wo, label="CountSketch++ w/o Preds", marker = markers[0])
            ax[j][0].fill_between(range(len(S_array)), u_vals_wo - u_std_wo, u_vals_wo + u_std_wo, alpha=0.2)
            ax[j][1].plot(w_vals_wo, label="CountSketch++ w/o Preds", marker = markers[0])
            ax[j][1].fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)

            for i, name in enumerate(labels):
                u_vals, u_std = mean_std_across_trials(uerrs[i][c])
                w_vals, w_std = mean_std_across_trials(werrs[i][c])

                ax[j][0].plot(u_vals, label=name, marker = markers[i+1])
                ax[j][0].fill_between(range(len(S_array)), u_vals - u_std, u_vals + u_std, alpha=0.2)
                ax[j][1].plot(w_vals, label=name, marker = markers[i+1])
                ax[j][1].fill_between(range(len(S_array)), w_vals - w_std, w_vals + w_std, alpha=0.2)
            

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
        plt.savefig(output_dir + "/all_in_one.png", bbox_inches='tight')
        plt.close()

        for j, c in enumerate(threshConsts):
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
                
            ax.set_xlabel('Sketch Size')
            ax.set_ylabel('Weighted Error')
            ax.set_xticks(range(len(S_array)))
            ax.set_xticklabels([str(s) for s in S_array])

            # Plot lines
            markers = get_markers()
            w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werrs_CountSketchPlusPlus_wo_preds_by_tau[c])
            ax.plot(w_vals_wo, label="CountSketch++ w/o Preds", marker = markers[0])
            ax.fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)

            werrs = list(self.trials_werrs_by_oracle.values())
            for i, name in enumerate(labels):
                u_vals, u_std = mean_std_across_trials(uerrs[i][c])
                w_vals, w_std = mean_std_across_trials(werrs[i][c])

                ax.plot(w_vals, label=name, marker = markers[i+1])
                ax.fill_between(range(len(S_array)), w_vals - w_std, w_vals + w_std, alpha=0.2)
           
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
            plt.savefig(output_dir + "/C_{}.png".format(c), bbox_inches='tight')
            plt.close()









    
