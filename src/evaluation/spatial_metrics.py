import numpy as np
import torch


def weighted_rmse(xr_groundtruth, xr_pred):
    weights = np.cos(np.deg2rad(xr_groundtruth.lat))
    wrmse = np.sqrt(((xr_groundtruth - xr_pred)**2).weighted(weights).mean(['lat', 'lon', 'time']))
    return wrmse.data.item()


def nll(data, mu, sigma):
    res = -0.5 * (np.log(2 * np.pi * (sigma**2)) + (data - mu)**2 / sigma**2)
    return res


def weighted_nll(xr_groundtruth, xr_mean, xr_stddev):
    weights = np.cos(np.deg2rad(xr_groundtruth.lat))
    xr_nll = nll(xr_groundtruth, xr_mean, xr_stddev)
    wnll = xr_nll.weighted(weights).mean(['lat', 'lon', 'time'])
    return wnll.data.item()


def compute_calib95_ICI(xr_groundtruth, xr_mean, xr_stddev):
    # Convert to torch tensors
    gt = torch.from_numpy(xr_groundtruth.data)
    dist = torch.distributions.Normal(loc=torch.from_numpy(xr_mean.data),
                                      scale=torch.from_numpy(xr_stddev.data))

    # Compute 95% calibration score
    lb, ub = dist.icdf(torch.tensor(0.025)), dist.icdf(torch.tensor(0.975))
    mask = (gt >= lb) & (gt <= ub)
    calib95 = mask.float().mean().item()

    # Compute integral calibration index
    cr_sizes = np.arange(0.05, 1.0, 0.05)
    calibs = []
    for size in cr_sizes:
        q_lb = (1 - float(size)) / 2
        q_ub = 1 - q_lb
        lb, ub = dist.icdf(torch.tensor(q_lb)), dist.icdf(torch.tensor(q_ub))
        mask = (gt >= lb) & (gt <= ub)
        calibs.append(mask.float().mean().item())
    ICI = np.abs(np.asarray(calibs) - cr_sizes).mean().item()
    return ICI, calib95
