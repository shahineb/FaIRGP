import os
import sys
import torch
import numpy as np
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scienceplots
plt.style.use('science')

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

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.plot(test_times, test_tas, color='cornflowerblue', lw=4, label=test_scenarios.names[0])
    ax.plot(test_times, test_tas_fair, color='tomato', ls='--', lw=5, label='FaIR')
    ax.plot(test_times, posterior_dist.mean, color='orange', lw=7, label='Posterior mean')
    ax.fill_between(test_times, posterior_lb, posterior_ub, alpha=0.3, color='orange', label='Confidence region')
    ax.tick_params(labelsize=16)
    ax.set_ylabel('Temperature anomaly (K)', fontsize=22)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=20)
    plt.tight_layout()
    return fig, ax


def plot_contourf_with_zonal_avg(field, title="", levels=20, fontsize=14):
    '''input field should be a 2D xarray.DataArray on a lat/lon grid.
        Make a filled contour plot of the field, and a line plot of the zonal mean
        Stolen from : https://brian-rose.github.io/ClimateLaboratoryBook/courseware/transient-cesm.html
    '''
    fig = plt.figure(figsize=(14, 6))
    nrows = 10
    ncols = 3
    mapax = plt.subplot2grid((nrows, ncols), (0, 0), colspan=ncols - 1, rowspan=nrows - 1, projection=ccrs.Robinson())
    barax = plt.subplot2grid((nrows, ncols), (nrows - 1, 0), colspan=ncols - 1)
    plotax = plt.subplot2grid((nrows, ncols), (0, ncols - 1), rowspan=nrows - 1)
    wrap_data, wrap_lon = add_cyclic_point(field.values, coord=field.lon, axis=field.dims.index('lon'))
    vmax = np.abs(wrap_data).max()
    cx = mapax.contourf(wrap_lon, field.lat, wrap_data, transform=ccrs.PlateCarree(),
                        levels=levels, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    mapax.set_global()
    mapax.coastlines()
    plt.colorbar(cx, cax=barax, orientation='horizontal')
    field_mean = field.mean(dim='lon')
    field_std = field.std(dim='lon')
    plotax.plot(field_mean, field.lat, '--.', lw=2)
    plotax.fill_betweenx(field.lat, field_mean - field_std, field_mean + field_std, alpha=0.3)
    plotax.set_ylabel('Latitude', fontsize=fontsize)
    plotax.set_xlabel(r'$Delta T$ (K)', fontsize=fontsize)
    plotax.grid()
    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    return fig, (mapax, plotax, barax), cx


def plot_contourf_on_ax(field, fig, ax, levels=20, cmap='RdBu_r', vmax=None, vmin=None, title="", fontsize=14, colorbar=False):
    '''input field should be a 2D xarray.DataArray on a lat/lon grid.
    '''
    wrap_data, wrap_lon = add_cyclic_point(field.values,
                                           coord=field.lon,
                                           axis=field.dims.index('lon'))
    if vmax is None:
        vmax = np.abs(wrap_data).max()
    if vmin is None:
        vmin = -vmax
    levels = np.linspace(vmin, vmax, levels)
    cx = ax.contourf(wrap_lon,
                     field.lat,
                     wrap_data,
                     levels=levels,
                     cmap=cmap,
                     transform=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    if colorbar:
        cax = ax.inset_axes((1.02, 0, 0.02, 1))
        cbar = fig.colorbar(cx, cax=cax)
        cbar.set_label(r'$\Delta T$ (K)', fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=fontsize)

    output = (fig, ax, cbar) if colorbar else (fig, ax)
    return output


def plot_tryptych(xr_prior_mean,
                  xr_correction,
                  xr_posterior_mean,
                  xr_posterior_ub,
                  xr_posterior_lb,
                  xr_groundtruth):
    fig, ax = plt.subplots(3, 4, figsize=(22, 9), subplot_kw={'projection': ccrs.Robinson()})

    for i in [0, 1, 3]:
        ax[0, i].remove()
        ax[2, i].remove()

    vmax = max(np.abs(xr_posterior_ub.data).max(), np.abs(xr_posterior_lb.data).max(), np.abs(xr_groundtruth.data).max())

    _, __, cbar_prior_mean = plot_contourf_on_ax(xr_prior_mean, fig, ax[1, 0], title="Prior mean", vmax=vmax, colorbar=True, fontsize=20)
    _, __, cbar_correction = plot_contourf_on_ax(xr_correction, fig, ax[1, 1], title="Posterior correction", colorbar=True, cmap='PiYG_r', fontsize=20)
    _, __, cbar_posterior_ub = plot_contourf_on_ax(xr_posterior_ub, fig, ax[0, 2], title="Posterior upper bound", vmax=vmax, colorbar=True, fontsize=20)
    _, __, cbar_posterior_mean = plot_contourf_on_ax(xr_posterior_mean, fig, ax[1, 2], title="Posterior mean", vmax=vmax, colorbar=True, fontsize=20)
    _, __, cbar_posterior_lb = plot_contourf_on_ax(xr_posterior_lb, fig, ax[2, 2], title="Posterior lower bound", vmax=vmax, colorbar=True, fontsize=20)
    _, __, cbar_groundtruth = plot_contourf_on_ax(xr_groundtruth, fig, ax[1, 3], title="Groundtruth", vmax=vmax, colorbar=True, fontsize=20)
    cbars = (cbar_prior_mean, cbar_correction, cbar_posterior_mean, cbar_posterior_lb, cbar_posterior_ub, cbar_groundtruth)

    plt.tight_layout()
    return fig, ax, cbars


def plot_timeserie_maps(xr_timeserie):
    n_time = len(xr_timeserie.time)
    fig, ax = plt.subplots(1, n_time, figsize=(5 * n_time, 10), subplot_kw={'projection': ccrs.Robinson()})

    vmax = np.abs(xr_timeserie.data).max()

    for i in range(n_time - 1):
        _ = plot_contourf_on_ax(field=xr_timeserie.isel(time=i),
                                fig=fig,
                                ax=ax[i],
                                title=int(xr_timeserie.time.data[i]),
                                vmax=vmax)
    _ = plot_contourf_on_ax(field=xr_timeserie.isel(time=-1),
                            fig=fig,
                            ax=ax[-1],
                            title=int(xr_timeserie.time.data[-1]),
                            vmax=vmax,
                            colorbar=True)
    plt.tight_layout()
    return fig, ax
