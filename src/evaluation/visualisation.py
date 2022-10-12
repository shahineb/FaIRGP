import os
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': False,
    'font.family': 'serif',
}
matplotlib.rcParams.update(params)

base_dir = os.path.join(os.getcwd(), '../..')
sys.path.append(base_dir)

from src.models.utils import compute_means


def colorbar(mappable):
    """
    Stolen from https://joseph-long.com/writing/colorbars/ (thank you!)
    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_scenario_prediction(posterior_dist, test_scenarios, model):
    test_times = test_scenarios.timesteps
    test_tas = test_scenarios.tas

    test_fair_means = compute_means(test_scenarios)
    test_tas_fair = torch.cat([v for v in test_fair_means.values()]).sum(dim=-1)

    posterior_lb, posterior_ub = posterior_dist.confidence_region()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(test_times, test_tas, color='cornflowerblue', label=test_scenarios.names[0])
    ax.plot(test_times, test_tas_fair, color='tomato', ls='--', lw=3, label='FaIR')
    ax.plot(test_times, posterior_dist.mean, color='orange', label='Posterior mean')
    ax.fill_between(test_times, posterior_lb, posterior_ub, alpha=0.5, color='orange', label='Confidence region')
    ax.set_ylabel('Temperature anomaly (K)')
    ax.grid(alpha=0.5)
    ax.legend()
    plt.tight_layout()
    return fig, ax
