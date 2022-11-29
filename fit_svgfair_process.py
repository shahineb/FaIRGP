"""
Description : Fits FaIR-contrained SVGP for spatial temperature response emulation

Usage: fit_svgfair_process.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
"""
import os
import yaml
import logging
from docopt import docopt
import tqdm
from collections import defaultdict
import numpy as np
import torch
from gpytorch import kernels, likelihoods
from src.preprocessing.spatial import make_data
from src.torchFaIR import FaIR
from src.models import ThermalBoxesSVGP
import src.models.utils_svgp as utils
from src.evaluation import dump_state_dict, dump_logs


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg)

    # Move needed tensors only to device
    data = migrate_to_device(data=data, device=device)

    # Instantiate model
    model = make_model(cfg=cfg, data=data).to(device)
    logging.info(f"{model}")

    # Fit model
    logging.info("\n Fitting model")
    model, logs = fit(model=model, data=data, cfg=cfg)

    # Dump model weights and logs
    dump(model=model, logs=logs, output_dir=args['--o'])
    logging.info(f"\n State dict dumped under {args['--o']}")


def migrate_to_device(data, device):
    # These are the only tensors needed on device to run this experiment
    data.scenarios._clear_cache()
    data.inducing_scenario._clear_cache()
    data = data._replace(scenarios=data.scenarios.to(device),
                         inducing_scenario=data.inducing_scenario.to(device),
                         S0=data.S0.to(device),
                         d_map=data.d_map.to(device),
                         q_map=data.q_map.to(device))
    return data


def make_model(cfg, data):
    # Instantiate backbone FaIR model and deactivate training
    forcing_pattern = np.ones((len(data.scenarios[0].lat), len(data.scenarios[0].lon)))  # uniform forcing pattern for now
    # with torch.no_grad():
    FaIR_model = FaIR(**data.fair_kwargs, forcing_pattern=forcing_pattern)
    for param in FaIR_model.parameters():
        param.requires_grad = False
    FaIR_model = FaIR_model.to(data.S0.device)

    # Instantiate kernel for GP prior over forcing
    kernel = kernels.MaternKernel(nu=1.5, ard_num_dims=4, active_dims=[1, 2, 3, 4])

    # Instantiate gaussian observation likelihood
    likelihood = likelihoods.GaussianLikelihood()

    # Instantiate FaIR-constrained SVGP
    model = ThermalBoxesSVGP(scenario_dataset=data.scenarios,
                             inducing_scenario=data.inducing_scenario,
                             kernel=kernel,
                             likelihood=likelihood,
                             FaIR_model=FaIR_model,
                             S0=data.S0,
                             q_map=data.q_map,
                             d_map=data.d_map)
    return model


def fit(model, data, cfg):
    # Set model in training mode
    model.train()

    # Retrieve useful values
    n_samples = data.scenarios.tas.numel()
    batch_size_time = cfg['training']['batch_size_time']
    batch_size_lat = cfg['training']['batch_size_lat']
    batch_size_lon = cfg['training']['batch_size_lon']
    batch_size = batch_size_time * batch_size_lat * batch_size_lon

    # Define optimiser and marginal likelihood objective
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    # Setup progress bar
    training_iter = tqdm.tqdm(range(cfg['training']['n_epochs']), desc='Iter')

    for epoch in training_iter:

        # Setup progress bar, display vars and record dictionnary
        batch_iter = tqdm.tqdm(range(n_samples // batch_size), desc='Batch')
        epoch_ell, epoch_kl, epoch_loss = 0, 0, 0
        logs = defaultdict(list)

        for i in batch_iter:
            # Sample batch
            batch = model.sample_batch(batch_size_time, batch_size_lat, batch_size_lon)
            scenario, time_idx, lat_idx, lon_idx, targets = batch

            # Zero out remaining gradients
            optimizer.zero_grad()

            # Compute variational posterior
            qf = model(scenario, time_idx, lat_idx, lon_idx, diag=True)

            # Compute ELBO loss
            ell, α, β, γ = utils.compute_ell(qf, targets, batch_size, model)
            kl_divergence = model.variational_strategy.kl_divergence().mul(batch_size / n_samples)
            loss = (kl_divergence - ell) / batch_size

            # Take gradient step
            loss.backward()
            optimizer.step()

            # Display updated epoch loss
            epoch_ell += ell.item()
            epoch_kl += kl_divergence.item()
            epoch_loss += loss.item()
            batch_iter.set_postfix_str(f"ELL {epoch_ell / (i + 1):e} | KL {epoch_kl / (i + 1):e} | Loss {epoch_loss / (i + 1):e}")

            # Record batch losses values
            logs['alpha'].append(α.detach().cpu().item())
            logs['beta'].append(β.detach().cpu().item())
            logs['gamma'].append(γ.detach().cpu().item())
            logs['ell'].append(ell.detach().cpu().item())
            logs['kl'].append(kl_divergence.detach().cpu().item())
            logs['elbo'].append(-loss.detach().cpu().item())

    return model, logs


def dump(model, logs, output_dir):
    dump_state_dict(model=model, output_dir=output_dir)
    dump_logs(logs=logs, output_dir=output_dir)


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Setup global variable for device
    if torch.cuda.is_available() and args['--device'].isdigit():
        device = torch.device(f"cuda:{args['--device']}")
    else:
        device = torch.device('cpu')

    # Run session
    main(args, cfg)
