from experiments.src.experiments.setup import ExperimentSetUp
from experiments.src.experiments.oracles import OracleCollection
from experiments.src.experiments.overall_comparison.exp_overall_comp import ExpOverallComp
from collections import defaultdict
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from experiments.config import  settings
import matplotlib.cm as cm
from experiments.src.utils import set_global_plot_style

class MultiRunExpOverallComp:
    def __init__(self):
        self.trials_uerr_by_oracle_p1 = defaultdict(lambda: defaultdict(list))
        self.trials_werr_by_oracle_p1 = defaultdict(lambda: defaultdict(list))

        self.trials_werr_by_oracle_p2 = defaultdict(list)
        self.trials_uerr_by_oracle_p2 = defaultdict(list)

        self.trials_werr_by_oracle_p3 = defaultdict(lambda: defaultdict(list))
        self.trials_uerr_by_oracle_p3 = defaultdict(lambda: defaultdict(list))

    def update(self, trial_run: ExpOverallComp):
        for oracle_name, v in trial_run.uerr_by_oracle_p1.items():
            for thres_const in v.keys():
                self.trials_uerr_by_oracle_p1[oracle_name][thres_const].append(v[thres_const])

        for oracle_name, v in trial_run.werr_by_oracle_p1.items():
            for thres_const in v.keys():
                self.trials_werr_by_oracle_p1[oracle_name][thres_const].append(v[thres_const])
        
        for oracle_name, v in trial_run.uerr_by_oracle_p2.items():
            self.trials_uerr_by_oracle_p2[oracle_name].append(v)

        for oracle_name, v in trial_run.werr_by_oracle_p2.items():
            self.trials_werr_by_oracle_p2[oracle_name].append(v)

        for oracle_name, v in trial_run.uerr_by_oracle_p3.items():
            for thres_const in v.keys():
                self.trials_uerr_by_oracle_p3[oracle_name][thres_const].append(v[thres_const])

        for oracle_name, v in trial_run.werr_by_oracle_p3.items():
            for thres_const in v.keys():
                self.trials_werr_by_oracle_p3[oracle_name][thres_const].append(v[thres_const])

    def run(self, setup_by_seed: dict[int,ExperimentSetUp], log_dirs: dict[int,list[str]]):

        for seed, setup in setup_by_seed.items():
            np.random.seed(seed)
            logging.info(f"\n\nStarting trial with seed {seed}...\n")
            
            trial_run = ExpOverallComp(log_dirs[seed])
            trial_run.run(setup=setup)
            self.update(trial_run)

        threshConsts = setup_by_seed[next(iter(setup_by_seed))].threshConsts
        S_array = setup_by_seed[next(iter(setup_by_seed))].S_array
        output_base_path = setup_by_seed[next(iter(setup_by_seed))].output_base_path
        start_time = setup_by_seed[next(iter(setup_by_seed))].start_time

        output_dir = os.path.join(output_base_path, "e4",  str(start_time).split('.')[0], "plots")
        os.makedirs(output_dir, exist_ok=True)
        self.aggregate_and_plot_trials_part_4(threshConsts, S_array, output_dir)

        logging.info("\n \n %d trials completed for experiment 4. ✅ \n", len(setup_by_seed))


    def aggregate_and_plot_trials_part_4(self, threshConsts, S_array, output_dir: str):
        set_global_plot_style()

        def mean_std_across_trials(list_of_lists):
            arr = np.array(list_of_lists)
            return np.mean(arr, axis=0), np.std(arr, axis=0)

        fig, ax = plt.subplots(len(threshConsts), 2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW* len(threshConsts)), constrained_layout=True)
        for i, ax_array in enumerate(ax):
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            ax_array[0].set_title('Unweighted Error (C={})'.format(threshConsts[i]))
            ax_array[1].set_title('Weighted Error; (C={})'.format(threshConsts[i]))
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(S_array)))
                ax_i.set_xticklabels([str(s) for s in S_array])

        markers = ["d", "<", "."]  

        # Plot lines for part 1
        labels_p1 = ["CountSketch++ + Space Savings (2 passes)", "CountSketch++ + Misra Gries (2 passes)", "CountSketch++ + perfect Oracle (2 passes)"]
        for i, oracle_name in enumerate(self.trials_uerr_by_oracle_p1.keys()):
            for j, c in enumerate(threshConsts):
                u_vals_wo, u_std_wo = mean_std_across_trials(self.trials_uerr_by_oracle_p1[oracle_name][c])
                w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werr_by_oracle_p1[oracle_name][c])
                ax[j][0].plot(u_vals_wo,  label=labels_p1[i], linestyle='-', marker=markers[i])
                ax[j][0].fill_between(range(len(S_array)), u_vals_wo - u_std_wo, u_vals_wo + u_std_wo, alpha=0.2)
                ax[j][1].plot(w_vals_wo,  label=labels_p1[i], linestyle='-', marker=markers[i])
                ax[j][1].fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)


        # Plot lines for part 2
        labels_p2 = ["Space Savings (1 pass)", "Misra Gries (1 pass)"]
        for i, oracle_name in enumerate(self.trials_uerr_by_oracle_p2.keys()):
            for j in range(len(threshConsts)):
                u_vals_wo, u_std_wo = mean_std_across_trials(self.trials_uerr_by_oracle_p2[oracle_name])
                w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werr_by_oracle_p2[oracle_name])
                
                ax[j][0].plot(u_vals_wo,  label=labels_p2[i], linestyle='-', marker=markers[i])
                ax[j][0].fill_between(range(len(S_array)), u_vals_wo - u_std_wo, u_vals_wo + u_std_wo, alpha=0.2)
                ax[j][1].plot(w_vals_wo,  label=labels_p2[i], linestyle='-', marker=markers[i])
                ax[j][1].fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)

        # Plot lines for part 3
        labels_p3 = ["Augm. Oracle: Space Savings (1 pass)", "Augm. Oracle: Misra Gries (1 pass)"]
        for i, oracle_name in enumerate(self.trials_uerr_by_oracle_p3.keys()):

            for j, c in enumerate(threshConsts):
                u_vals_wo, u_std_wo = mean_std_across_trials(self.trials_uerr_by_oracle_p3[oracle_name][c])
                w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werr_by_oracle_p3[oracle_name][c])
                ax[j][0].plot(u_vals_wo,  label=labels_p3[i], linestyle=':', marker=markers[i])
                ax[j][0].fill_between(range(len(S_array)), u_vals_wo - u_std_wo, u_vals_wo + u_std_wo, alpha=0.2)
                ax[j][1].plot(w_vals_wo,  label=labels_p3[i], linestyle=':', marker=markers[i])
                ax[j][1].fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)


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
        plt.savefig(output_dir + "/all_in_one.png", bbox_inches='tight')
        plt.close()

        for j, c in enumerate(threshConsts):
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
                
            ax.set_xlabel('Sketch Size')
            ax.set_ylabel('Weighted Error')
            ax.set_xticks(range(len(S_array)))
            ax.set_xticklabels([str(s) for s in S_array])

            markers = ["d", "<", "."]  
            colors = [cm.get_cmap('Paired')(i) for i in [0,3,9]]

            # Plot lines for part 1
            labels_p1 = ["CountSketch++ + Space Savings (2 passes)", "CountSketch++ + Misra Gries (2 passes)", "CountSketch++ + perfect Oracle (2 passes)"]
            for i, oracle_name in enumerate(self.trials_uerr_by_oracle_p1.keys()):
                w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werr_by_oracle_p1[oracle_name][c])
                ax.plot(w_vals_wo,  label=labels_p1[i], linestyle='-', marker=markers[i])
                ax.fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)


            # Plot lines for part 2
            labels_p2 = ["Space Savings (1 pass)", "Misra Gries (1 pass)"]
            for i, oracle_name in enumerate(self.trials_uerr_by_oracle_p2.keys()):
                w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werr_by_oracle_p2[oracle_name])
                ax.plot(w_vals_wo,  label=labels_p2[i], linestyle='-', marker=markers[i])
                ax.fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)

            # Plot lines for part 3
            labels_p3 = ["Augm. Oracle: Space Savings (1 pass)", "Augm. Oracle: Misra Gries (1 pass)"]
            for i, oracle_name in enumerate(self.trials_uerr_by_oracle_p3.keys()):
                w_vals_wo, w_std_wo = mean_std_across_trials(self.trials_werr_by_oracle_p3[oracle_name][c])
                ax.plot(w_vals_wo,  label=labels_p3[i], linestyle=':', marker=markers[i])
                ax.fill_between(range(len(S_array)), w_vals_wo - w_std_wo, w_vals_wo + w_std_wo, alpha=0.2)


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

