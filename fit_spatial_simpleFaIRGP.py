"""
Description : Fits FaIR-contrained GP for global temperature response emulation

Usage: fit_spatial_simpleFaIRGP.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
import torch
import numpy as np
from gpytorch import kernels, mlls
from src.preprocessing.spatial import make_data, get_fair_params
from src.torchFaIR import FaIR
from src.models import SimpleMultiThermalBoxesGP
from src.likelihoods import InternalVariability
from src.evaluation import dump_state_dict


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
    model = fit(model=model, data=data, cfg=cfg)

    # Dump model weights
    dump_state_dict(model=model, output_dir=args['--o'])
    logging.info(f"\n State dict dumped under {args['--o']}")


def migrate_to_device(data, device):
    # These are the only tensors needed on device to run this experiment
    return data


def make_model(cfg, data):
    # Instantiate FaIR model
    nlat, nlon = len(data.scenarios[0].lat), len(data.scenarios[0].lon)
    base_kwargs = get_fair_params()
    forcing_pattern = np.ones((nlat, nlon))
    with torch.no_grad():
        FaIR_model = FaIR(**base_kwargs, forcing_pattern=forcing_pattern, requires_grad=False)

    # Instantiate kernel for GP prior over forcing
    kernel = kernels.ScaleKernel(kernels.MaternKernel(nu=1.5, ard_num_dims=4, active_dims=[1, 2, 3, 4]))

    # Instantiate internal variability likelihood module
    likelihood = InternalVariability(q=data.fair_kwargs['q'], d=data.fair_kwargs['d'])

    # Instantiate FaIR-constrained GP
    model = SimpleMultiThermalBoxesGP(scenario_dataset=data.scenarios,
                                      kernel=kernel,
                                      FaIR_model=FaIR_model,
                                      likelihood=likelihood,
                                      S0=data.S0,
                                      q_map=data.q_map,
                                      d_map=data.d_map)
    return model


def fit(model, data, cfg):
    # Set model in training mode
    model.train()
    flattened_targets = model.train_targets.view(len(model.train_scenarios.timesteps), -1).T

    # Define optimiser and marginal likelihood objective
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    mll = mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Setup progress bar
    training_iter = tqdm.tqdm(range(cfg['training']['n_epochs']), desc='Iter')

    for i in training_iter:
        optimizer.zero_grad()
        output = model.train_prior_dist()
        loss = -mll(output, flattened_targets).mean()
        loss.backward()
        optimizer.step()
        training_iter.set_postfix_str(f"LL = {-loss.item()}")
    return model


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
