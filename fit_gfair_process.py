"""
Description : Fits FaIR-contrained GP for global temperature response emulation

Usage: fit_gfair_process.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
from gpytorch import kernels, likelihoods, mlls
from src.preprocessing.glob import make_data
from src.models import ThermalBoxesGP
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
    # Instantiate kernel for GP prior over forcing
    forcing_kernel = kernels.MaternKernel(nu=1.5, ard_num_dims=4, active_dims=[1, 2, 3, 4])
    ou_kernel = kernels.ScaleKernel(kernels.MaternKernel(nu=0.5, ard_num_dims=1, active_dims=[0]))
    kernel = forcing_kernel + ou_kernel

    # Instantiate gaussian observation likelihood
    likelihood = likelihoods.GaussianLikelihood()

    # Instantiate FaIR-constrained GP
    model = ThermalBoxesGP(scenario_dataset=data.scenarios,
                           kernel=kernel,
                           likelihood=likelihood,
                           q=data.fair_kwargs['q'],
                           d=data.fair_kwargs['d'])
    return model


def fit(model, data, cfg):
    # Set model in training mode
    model.train()

    # Define optimiser and marginal likelihood objective
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    mll = mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Setup progress bar
    training_iter = tqdm.tqdm(range(cfg['training']['n_epochs']), desc='Iter')

    for i in training_iter:
        optimizer.zero_grad()
        output = model()
        loss = -mll(output, model.train_targets)
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
