# plot_config.py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler
import pandas as pd
import matplotlib.pyplot as plt
from experiments.config import settings
from typing import Literal
from experiments.src.loader import loadAOL

cmap = cm.get_cmap('tab10')
color_indices = [0,1,2,3,4,5,6,7]
#cmap = cm.get_cmap('Paired')
#color_indices = [0,1,2,3,8,9,6]  # Adjusted to match the colors used in the original code
colors = [cmap(i) for i in color_indices]
custom_colors = colors

def set_global_plot_style():
    plt.rcParams['axes.prop_cycle'] = cycler(color=custom_colors)

    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['axes.labelsize'] = 14


    # Font size settings
    plt.rcParams['font.size'] = 12          # General font size
    plt.rcParams['axes.titlesize'] = 14     # Title font size
    plt.rcParams['axes.labelsize'] = 12     # X and Y label font size
    plt.rcParams['xtick.labelsize'] = 12    # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 12    # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 12    # Legend font size
    plt.rcParams['figure.titlesize'] = 14   # Figure title (suptitle)


def get_markers(): 
    return ['o', "d", "<", "P", "x", "*", "|",]


def set_plot_style_part_4():

    plt.rcParams['axes.prop_cycle'] = cycler(color=[custom_colors[i] for i in [1,2,7]])
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams
    