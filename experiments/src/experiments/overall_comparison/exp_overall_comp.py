""" Experiment 4: Comparing 2 best settings of augmented oracles in 1 pass setting with best combinations in 2 pass setting
--------------------------------------

This experiment compares the performance of the methods that proved the most promissing in the first part and in the third part. 

For the first part (2 passes), these are: 
- CountSketch++ + perfect Oracle 
- CountSketch++ + Space Savings (best approximate oracle)
- CountSketch++ + Misra Gries (best approximate oracle)

For the second part (1 pass), these are: 
- Space Savings 
- Misra Gries

For the third part (1 pass), these are:
- Space Savings + (separate) CountSketch++
- Misra Gries + (separate) CountSketch++
"""
import matplotlib.cm as cm
from collections import defaultdict
import logging
import os 
from experiments.config import  settings
import yaml
import matplotlib.pyplot as plt
from experiments.src.utils import set_global_plot_style
from experiments.src.experiments import ExperimentSetUp, OracleCollection, Experiment


class ExpOverallComp(Experiment):
    def __init__(self, log_dirs: list[str]):
        self.logs_dir_1 = log_dirs[0]
        self.logs_dir_2 = log_dirs[1]
        self.logs_dir_3 = log_dirs[2]
        self.uerr_by_oracle_p1 = defaultdict(lambda: defaultdict(list))
        self.werr_by_oracle_p1 = defaultdict(lambda: defaultdict(list))

        self.werr_by_oracle_p2 = defaultdict(list)
        self.uerr_by_oracle_p2 = defaultdict(list)

        self.werr_by_oracle_p3 = defaultdict(lambda: defaultdict(list))
        self.uerr_by_oracle_p3 = defaultdict(lambda: defaultdict(list))


    def run(self, setup: ExperimentSetUp):
        logging.info("Running Overall Comparison with setup: %s", setup.get_params())

        # Load logs from previous parts
        # Part 1
        with open(os.path.join(self.logs_dir_1, 'errors.yml'), 'r') as file:
            logs_p1 = yaml.safe_load(file)
            self.uerr_by_oracle_p1 = {"SpaceSavings": logs_p1["unweighted errors"]["SpaceSavings"], 
                                      "MisraGries": logs_p1["unweighted errors"]["MisraGries"],
                                      "perfect": logs_p1["unweighted errors"]["perfect"]}
            
            self.werr_by_oracle_p1 = {"SpaceSavings": logs_p1["weighted errors"]["SpaceSavings"],
                                       "MisraGries": logs_p1["weighted errors"]["MisraGries"],
                                       "perfect": logs_p1["weighted errors"]["perfect"]}
            
        # Part 2
        with open(os.path.join(self.logs_dir_2, 'errors.yml'), 'r') as file:
            logs_p2 = yaml.safe_load(file)
            self.uerr_by_oracle_p2 = {"SpaceSavings": logs_p2['unweighted errors (oracle estimates + outputting 0)']["SpaceSavings"],
                                       "MisraGries": logs_p2['unweighted errors (oracle estimates + outputting 0)']["MisraGries"]}
            
            self.werr_by_oracle_p2 = {"SpaceSavings": logs_p2['weighted errors (oracle estimates + outputting 0)']["SpaceSavings"],
                                       "MisraGries": logs_p2['weighted errors (oracle estimates + outputting 0)']["MisraGries"]}
        
        # Part 3
        with open(os.path.join(self.logs_dir_3, 'errors.yml'), 'r') as file:
            logs_p3 = yaml.safe_load(file)
            self.uerr_by_oracle_p3 = {"SpaceSavings": logs_p3['unweighted errors (oracle + sketch)']["SpaceSavings"],
                                       "MisraGries": logs_p3['unweighted errors (oracle + sketch)']["MisraGries"]}
            
            self.werr_by_oracle_p3 = {"SpaceSavings": logs_p3['weighted errors (oracle + sketch)']["SpaceSavings"],
                                       "MisraGries": logs_p3['weighted errors (oracle + sketch)']["MisraGries"]}

        output_path = os.path.join(setup.output_base_path, "e4", str(setup.start_time).split('.')[0], str(setup.seed))
        output_path_plots = os.path.join(output_path, "plots")
        os.makedirs(output_path_plots, exist_ok=True)
        self.write_logs(setup=setup, oracles=None, output_path=output_path)
        self.plot_single_run(setup=setup, output_path=output_path_plots) 
        logging.info("\n ################## Experiment Overall Comparison completed successfully for seed %d. ✅ ################## \n", setup.seed)


    def write_logs(self, setup: ExperimentSetUp, oracles: OracleCollection | None, output_path: str):
        errs = {
            'unweighted errors (oracle + sketch, part 1)': dict({k: dict(v) for k,v in self.uerr_by_oracle_p1.items()}),
            'weighted errors (oracle + sketch, part 1)': dict(self.werr_by_oracle_p1),
            'unweighted errors (oracle estimates + outputting 0, part 2)': dict(self.uerr_by_oracle_p2),
            'weighted errors (oracle estimates + outputting 0, part 2)': dict(self.werr_by_oracle_p2),
            'unweighted errors (oracle + sketch, part 3)': dict({k: dict(v) for k,v in self.uerr_by_oracle_p3.items()}),
            'weighted errors (oracle + sketch, part 3)': dict({k: dict(v) for k,v in self.werr_by_oracle_p3.items()})
        }
        if oracles:
            with open(output_path + '/oracles.yml', 'w') as outfile:
                yaml.dump(oracles.get_oracles(), outfile, default_flow_style=False, sort_keys=False)

        super().write_logs(setup=setup, errs=errs, output_path=output_path)


    def plot_single_run(self, setup: ExperimentSetUp, output_path: str):
        set_global_plot_style()
        
        fig, ax = plt.subplots(len(setup.threshConsts), 2, figsize=(settings.FIG_WIDTH_PER_COL, settings.FIG_HEIGHT_PER_ROW* len(setup.threshConsts)), constrained_layout=True)
        for i, ax_array in enumerate(ax):
            ax_array[0].set_ylabel('Unweighted Error')
            ax_array[1].set_ylabel('Weighted Error')
            ax_array[0].set_title('Unweighted Error (C={})'.format(setup.threshConsts[i]))
            ax_array[1].set_title('Weighted Error; (C={})'.format(setup.threshConsts[i]))
            for ax_i in ax_array:
                ax_i.set_xlabel('Sketch Size')
                ax_i.set_xticks(range(len(setup.S_array)))
                ax_i.set_xticklabels([str(s) for s in setup.S_array])

        markers = ["d", "<", "."]  

        # Plot lines for part 1
        labels_p1 = ["CountSketch++ + Space Savings (2 passes)", "CountSketch++ + Misra Gries (2 passes)", "CountSketch++ + perfect Oracle (2 passes)"]
        for i, oracle_name in enumerate(self.uerr_by_oracle_p1.keys()):
            for j, c in enumerate(setup.threshConsts):
                ax[j, 0].plot(setup.S_array, self.uerr_by_oracle_p1[oracle_name][c], label=labels_p1[i], linestyle='-', marker=markers[i])
                ax[j, 1].plot(setup.S_array, self.werr_by_oracle_p1[oracle_name][c], label=labels_p1[i], linestyle='-', marker=markers[i])

        # Plot lines for part 2
        labels_p2 = ["Space Savings (1 pass)", "Misra Gries (1 pass)"]
        for i, oracle_name in enumerate(self.uerr_by_oracle_p2.keys()):
            for j in range(len(setup.threshConsts)):
                ax[j, 0].plot(setup.S_array, self.uerr_by_oracle_p2[oracle_name], label=labels_p2[i], linestyle='--', marker=markers[i])
                ax[j, 1].plot(setup.S_array, self.werr_by_oracle_p2[oracle_name], label=labels_p2[i], linestyle='--', marker=markers[i])

        # Plot lines for part 3
        labels_p3 = ["Augm. Oracle: Space Savings (1 pass)", "Augm. Oracle: Misra Gries (1 pass)"]
        for i, oracle_name in enumerate(self.uerr_by_oracle_p3.keys()):
            for j, c in enumerate(setup.threshConsts):
                ax[j, 0].plot(setup.S_array, self.uerr_by_oracle_p3[oracle_name][c], label=labels_p3[i], linestyle=':', marker= markers[i])
                ax[j, 1].plot(setup.S_array, self.werr_by_oracle_p3[oracle_name][c], label=labels_p3[i], linestyle=':', marker= markers[i])

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
            ax.set_ylabel('Weighted Error')
            ax.set_xlabel('Sketch Size')
            ax.set_xticks(range(len(setup.S_array)))
            ax.set_xticklabels([str(s) for s in setup.S_array])

            # Plot lines
            for i, oracle_name in enumerate(self.uerr_by_oracle_p1.keys()):
                ax.plot(setup.S_array, self.werr_by_oracle_p1[oracle_name][c], label=labels_p1[i], linestyle='-', marker=markers[i])

            # Plot lines for part 2
            for i, oracle_name in enumerate(self.uerr_by_oracle_p2.keys()):
                ax.plot(setup.S_array, self.werr_by_oracle_p2[oracle_name], label=labels_p2[i], linestyle='--', marker=markers[i])

            # Plot lines for part 3
            labels_p3 = ["Augm. Oracle: Space Savings (1 pass)", "Augm. Oracle: Misra Gries (1 pass)"]
            for i, oracle_name in enumerate(self.uerr_by_oracle_p3.keys()):
                ax.plot(setup.S_array, self.werr_by_oracle_p3[oracle_name][c], label=labels_p3[i], linestyle=':', marker= markers[i])


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
        



