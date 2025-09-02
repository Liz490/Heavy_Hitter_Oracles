from experiments.src.experiments.setup import ExperimentSetUp
from experiments.src.experiments.oracles import OracleCollection
from experiments.src.experiments.part_1.single_exps_part_1 import ExperimentPart1
from collections import defaultdict
import logging
import numpy as np
from experiments.config import  settings
import matplotlib.pyplot as plt
from experiments.src.utils import get_markers, set_global_plot_style
import os
from experiments.src.utils import set_global_plot_style

class MultiRunExperimentPart1:
    def __init__(self):
        self.trials_uerrs_by_oracle_CountSketchPlusPlus = defaultdict(lambda: defaultdict(list))
        self.trials_werrs_by_oracle_CountSketchPlusPlus = defaultdict(lambda: defaultdict(list))
        self.trials_uerrs_CountSketchPlusPlus_wo_oracle = defaultdict(list)
        self.trials_werrs_CountSketchPlusPlus_wo_oracle = defaultdict(list)
        self.trials_uerrs_by_oracle_0 = defaultdict(list)
        self.trials_werrs_by_oracle_0 = defaultdict(list)


    def update(self, trial_run: ExperimentPart1):
        """
        Update the aggregated results with the results from a single trial.
        
        Args:
        ----------
            trial_exp (Experiment1): The Experiment1 instance containing results from a single trial.
        """
        for oracle_name, v in trial_run.uerrs_by_oracle_CountSketchPlusPlus.items():
            for thres_const in v.keys():
                self.trials_uerrs_by_oracle_CountSketchPlusPlus[oracle_name][thres_const].append(v[thres_const])

        for oracle_name, v in trial_run.werrs_by_oracle_CountSketchPlusPlus.items():
            for thres_const in v.keys():
                self.trials_werrs_by_oracle_CountSketchPlusPlus[oracle_name][thres_const].append(v[thres_const])

        for thres_const, v in trial_run.uerrs_CountSketchPlusPlus_wo_oracle.items():
            self.trials_uerrs_CountSketchPlusPlus_wo_oracle[thres_const].append(v)

        for thres_const, v in trial_run.werrs_CountSketchPlusPlus_wo_oracle.items():
            self.trials_werrs_CountSketchPlusPlus_wo_oracle[thres_const].append(v)

        for oracle_name, v in trial_run.uerrs_by_oracle_0.items():
            self.trials_uerrs_by_oracle_0[oracle_name].append(v)

        for oracle_name, v in trial_run.werrs_by_oracle_0.items():
            self.trials_werrs_by_oracle_0[oracle_name].append(v)


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
            trial_run = ExperimentPart1()
            output_path = trial_run.run(setup=setup, oracles=oracles)
            output_paths_by_seed[seed] = output_path

            self.update(trial_run)
        
        threshConsts = setup_by_seed[next(iter(setup_by_seed))].threshConsts
        S_array = setup_by_seed[next(iter(setup_by_seed))].S_array
        output_base_path = setup_by_seed[next(iter(setup_by_seed))].output_base_path
        start_time = setup_by_seed[next(iter(setup_by_seed))].start_time

        output_dir = os.path.join(output_base_path, "e1",  str(start_time).split('.')[0])
        self.aggregate_and_plot_trials_part_1(threshConsts, S_array, output_dir)

        logging.info("\n \n %d trials completed for experiment 1. ✅ \n", len(setup_by_seed))
        return output_paths_by_seed


    def aggregate_and_plot_trials_part_1(self, threshConsts, S_array, output_dir: str):
        set_global_plot_style()
        self.multi_trial_plot_oracles_plus_CountSketchPlusPlus_by_tau(threshConsts, S_array, output_dir + "/oracles_plus_CountSketchPlusPlus_by_tau")
        self.multi_trial_plot_CountSketchPlusPlus_vs_0_by_oracle(threshConsts, S_array, output_dir + "/CountSketchPlusPlus_vs_0_by_oracle")


    def multi_trial_plot_CountSketchPlusPlus_vs_0_by_oracle(self, threshConsts, S_array, output_path: str):

        def mean_std_across_trials(list_of_lists):
            arr = np.array(list_of_lists)
            return np.mean(arr, axis=0), np.std(arr, axis=0) 
        

        os.makedirs(output_path, exist_ok=True)
        labels = [f"CountSketch++ (C={c})" for c in threshConsts]
        oracle_names = list(self.trials_uerrs_by_oracle_CountSketchPlusPlus.keys())

        fig, ax = plt.subplots(len(oracle_names),2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW*len(oracle_names)), constrained_layout=True)
        for ax_array in ax:
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(S_array)))
                ax_i.set_xticklabels(S_array)

        for oracle_name, ax_i in zip(oracle_names, ax[:,0]):
            ax_i.set_title('Unweighted Error; Oracle: {}'.format(oracle_name))

        for oracle_name, ax_i in zip(oracle_names, ax[:,1]):
            ax_i.set_title('Weighted Error; Oracle: {}'.format(oracle_name))


         # Plot lines
        markers = get_markers()
        for j, name in enumerate(oracle_names):
            u_vals_wo, u_std_wo = mean_std_across_trials(self.trials_uerrs_by_oracle_0[name])
            w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werrs_by_oracle_0[name])

            ax[j][0].plot(u_vals_wo, label="Output 0 for Non-Top-k", marker = markers[0])
            ax[j][0].fill_between(range(len(S_array)), u_vals_wo - u_std_wo, u_vals_wo + u_std_wo, alpha=0.2)

            ax[j][1].plot(w_vals_wo, label="Output 0 for Non-Top-k", marker = markers[0])
            ax[j][1].fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)
        
            for i, c in enumerate(threshConsts):
                u_vals, u_std = mean_std_across_trials(self.trials_uerrs_by_oracle_CountSketchPlusPlus[name][c])
                w_vals, w_std = mean_std_across_trials(self.trials_werrs_by_oracle_CountSketchPlusPlus[name][c])

                ax[j][0].plot(u_vals, marker = markers[i+1],label=labels[i])
                ax[j][0].fill_between(range(len(S_array)), u_vals - u_std, u_vals + u_std, alpha=0.2)

                ax[j][1].plot(w_vals, marker = markers[i+1],label=labels[i])
                ax[j][1].fill_between(range(len(S_array)), w_vals - w_std, w_vals + w_std, alpha=0.2)


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
            ax.set_xticks(range(len(S_array)))
            ax.set_xticklabels([str(s) for s in S_array])


            # Plot lines
            markers = get_markers()
            w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werrs_by_oracle_0[oracle_name])

            ax.plot(w_vals_wo, label="Output 0 for Non-Top-k", marker = markers[0])
            ax.fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)

            for j, c in enumerate(threshConsts):
                w_vals, w_std = mean_std_across_trials(self.trials_werrs_by_oracle_CountSketchPlusPlus[oracle_name][c])

                ax.plot(w_vals, marker = markers[j+1],label=labels[j])
                ax.fill_between(range(len(S_array)), w_vals - w_std, w_vals + w_std, alpha=0.2)

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

        

        

    def multi_trial_plot_oracles_plus_CountSketchPlusPlus_by_tau(self, threshConsts, S_array, output_path: str):

        def mean_std_across_trials(list_of_lists):
            arr = np.array(list_of_lists)
            return np.mean(arr, axis=0), np.std(arr, axis=0)
        
        os.mkdir(output_path)
        labels = ["CountSketch++ + Space Savings Oracle", "CountSketch++ + Misra Gries Oracle",  "CountSketch++ + Count Sketch Oracle", "CountSketch++ + Count Min Oracle", "CountSketch++ + CountSketch++ Oracle", "CountSketch++ + Perfect Oracle", ]
        oracle_names = list(self.trials_uerrs_by_oracle_CountSketchPlusPlus.keys())
        fig, ax = plt.subplots(len(threshConsts), 2, figsize=(settings.FIG_WIDTH_PER_COL, settings. FIG_HEIGHT_PER_ROW* len(threshConsts)), constrained_layout=True)
        
        for i, c in enumerate(threshConsts):
            ax[i][0].set_title(f'Unweighted Error (C={c})')
            ax[i][1].set_title(f'Weighted Error (C={c})')
            ax[i][0].set_ylabel('Unweighted Error')
            ax[i][1].set_ylabel('Weighted Error')
            for a in ax[i]:
                a.set_xlabel("Sketch Size")
                a.set_xticks(range(len(S_array)))
                a.set_xticklabels(S_array)
                

        # Plot lines
        markers = get_markers()
        
        for j, c in enumerate(threshConsts):
            u_vals_wo, u_std_wo = mean_std_across_trials(self.trials_uerrs_CountSketchPlusPlus_wo_oracle[c])
            w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werrs_CountSketchPlusPlus_wo_oracle[c])

            ax[j][0].plot(u_vals_wo, label="CountSketch++ w/o Oracle", marker = markers[0])
            ax[j][0].fill_between(range(len(S_array)), u_vals_wo - u_std_wo, u_vals_wo + u_std_wo, alpha=0.2)

            ax[j][1].plot(w_vals_wo, label="CountSketch++ w/o Oracle", marker = markers[0])
            ax[j][1].fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)
        
        for i, name in enumerate(oracle_names):
            for j, c in enumerate(threshConsts):
                u_vals, u_std = mean_std_across_trials(self.trials_uerrs_by_oracle_CountSketchPlusPlus[name][c])
                w_vals, w_std = mean_std_across_trials(self.trials_werrs_by_oracle_CountSketchPlusPlus[name][c])

                ax[j][0].plot(u_vals, marker = markers[i+1],label=labels[i])
                ax[j][0].fill_between(range(len(S_array)), u_vals - u_std, u_vals + u_std, alpha=0.2)

                ax[j][1].plot(w_vals, marker = markers[i+1],label=labels[i])
                ax[j][1].fill_between(range(len(S_array)), w_vals - w_std, w_vals + w_std, alpha=0.2)

        handles, leg_labels = ax[0][0].get_legend_handles_labels()
        legend = fig.legend(handles, leg_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
        legend.set_alignment("left")
        plt.tight_layout(rect=(0.0, 0.1, 1.0, 1.))
        plt.savefig(os.path.join(output_path, "all_in_one.png"), bbox_inches="tight")
        plt.close(fig)

        for j, c in enumerate(threshConsts):
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
                
            ax.set_xlabel('Sketch Size')
            ax.set_ylabel('Weighted Error')
            ax.set_xticks(range(len(S_array)))
            ax.set_xticklabels([str(s) for s in S_array])
            # Plot lines
            markers = get_markers()
            w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werrs_CountSketchPlusPlus_wo_oracle[c])

            ax.plot(w_vals_wo, label="CountSketch++ w/o Oracle", marker = markers[0])
            ax.fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)

            for i, oracle_name in enumerate(oracle_names):
                w_vals, w_std = mean_std_across_trials(self.trials_werrs_by_oracle_CountSketchPlusPlus[oracle_name][c])

                ax.plot(w_vals, marker = markers[i+1],label=labels[i])
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
            plt.savefig(output_path + "/C_{}.png".format(c), bbox_inches='tight')
            plt.close()




