import torch
import numpy as np
from scipy.stats import pearsonr


def compute_scores(posterior_dist, test_data):
    """Computes prediction scores
    Args:
        posterior_dist (torch.distributions.Distribution): (time,)
        test_data (ScenarioDataset)
    Returns:
        type: Description of returned object.
    """
    # Extract posterior mean prediction
    posterior_mean = posterior_dist.mean.cpu()

    # Compute metrics over all predictions
    scores_deterministic = compute_deterministic_metrics(posterior_mean, test_data.tas)
    scores_probabilistic = compute_probabilistic_metrics(posterior_dist, test_data.tas)

    # Encapsulate scores into output dictionnary
    output = {**scores_deterministic, **scores_probabilistic}
    return output


def compute_deterministic_metrics(prediction, groundtruth):
    """Compute deterministic metrics between posterior mean and groundtruth

    Args:
        prediction (torch.Tensor): (time,)
        groundtruth (torch.Tensor): (time,)

    Returns:
        type: dict[float]
    """
    # Compute raw distances metrics
    difference = prediction.sub(groundtruth)
    mean_bias = difference.mean()
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()

    # Compute spearman correlation
    corr = spearman_correlation(prediction.flatten(), groundtruth.flatten())

    # Encapsulate results in output dictionnary
    output = {'mb': mean_bias.item(),
              'rmse': rmse.item(),
              'mae': mae.item(),
              'corr': corr}
    return output


def compute_probabilistic_metrics(predicted_dist, groundtruth):
    """Computes probabilistic metrics between posterior distribution and groundtruth

    Args:
        posterior_dist (torch.distributions.Distribution): (time,)
        groundtruth (torch.Tensor): (time,)

    Returns:
        type: dict[float]
    """
    # Create normal distribution vector
    pointwise_predicted_dict = torch.distributions.Normal(loc=predicted_dist.mean, scale=predicted_dist.stddev)

    # Compute average LL of groundtruth
    ll = pointwise_predicted_dict.log_prob(groundtruth).mean()

    # Compute 95% calibration score
    lb, ub = pointwise_predicted_dict.icdf(torch.tensor(0.025)), pointwise_predicted_dict.icdf(torch.tensor(0.975))
    mask = (groundtruth >= lb) & (groundtruth <= ub)
    calib95 = mask.float().mean()

    # Compute integral calibration index
    confidence_region_sizes = np.arange(0.05, 1.0, 0.05)
    calibs = []
    for size in confidence_region_sizes:
        q_lb = (1 - float(size)) / 2
        q_ub = 1 - q_lb
        lb, ub = pointwise_predicted_dict.icdf(torch.tensor(q_lb)), pointwise_predicted_dict.icdf(torch.tensor(q_ub))
        mask = (groundtruth >= lb) & (groundtruth <= ub)
        calibs.append(mask.float().mean().item())
    ICI = np.abs(np.asarray(calibs) - confidence_region_sizes).mean()

    # Encapsulate results in output dictionnary
    output = {'ll': ll.item(),
              'calib95': calib95.item(),
              'ICI': ICI.item()}
    return output


def spearman_correlation(x, y):
    """Computes Spearman Correlation between x and y
    Args:
        x (torch.Tensor)
        y (torch.Tensor)
    Returns:
        type: torch.Tensor
    """
    x_std = (x - x.mean()) / x.std()
    y_std = (y - y.mean()) / y.std()
    corr = float(pearsonr(x_std.numpy(), y_std.numpy())[0])
    return corr
