"""
Description : Runs inference with FaIR-contrained SVGP for spatial temperature response emulation

Usage: predict_svgfair_process.py  [options] --model=<model_output_dir> --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --model=<path_to_model_dir>      Path to model to use. Output of fit_svgfair_process.py.
  --o=<output_dir>                 Output directory.
  --plot                           Outputs plots.
  --device=<device_index>          Device to use [default: cpu]
"""
import os
import yaml
import logging
from docopt import docopt
from collections import namedtuple
import torch
import xarray as xr
import matplotlib.pyplot as plt
import fit_svgfair_process as fit_svgfair
from src.preprocessing.spatial.preprocess_data import load_emissions_dataset, load_response_dataset, make_scenario
import src.evaluation.spatial_metrics as metrics
import src.evaluation.visualisation as vis


def main(args, cfg, model_cfg):
    # Load trained model
    logging.info("Loading model")
    model = load_model(args=args, model_cfg=model_cfg)

    # Load prediction scenario
    logging.info("Loading test scenario")
    test_scenario = load_test_scenario(cfg=cfg)

    # Run prediction
    logging.info("Running inference")
    pred = predict(model=model, scenario=test_scenario, cfg=cfg)

    # Compute metrics
    scores = compute_scores(pred=pred)
    dump_path = os.path.join(args['--o'], 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(scores, f)
    logging.info(f"Dumped scores under {dump_path}")

    # Plot figures
    if args['--plot']:
        logging.info("Generating plots")
        dump_plots(pred=pred, cfg=cfg, output_dir=args['--o'])
        logging.info("Dumped plots")


def load_model(args, model_cfg):
    # Instantiate training set and model
    data = fit_svgfair.make_data(cfg=model_cfg)
    data = fit_svgfair.migrate_to_device(data=data, device=device)
    model = fit_svgfair.make_model(cfg=model_cfg, data=data).to(device)

    # Load model weights
    logging.info("Loading state dict")
    state_dict = torch.load(os.path.join(args['--model'], 'state_dict.pt'), map_location=device)
    model.load_state_dict(state_dict)
    return model


def load_test_scenario(cfg):
    # Load emissions and temperature response xarrays
    key = cfg['evaluation']['key']
    inputs_filepaths = {key: os.path.join(cfg['evaluation']['dirpath'], f"inputs_{key}.nc")}
    outputs_filepaths = {key: os.path.join(cfg['evaluation']['dirpath'], f"outputs_{key}.nc")}
    input_xarrays = {key: load_emissions_dataset(filepath) for (key, filepath) in inputs_filepaths.items()}
    output_xarrays = {key: load_response_dataset(filepath) for (key, filepath) in outputs_filepaths.items()}

    if cfg['evaluation']['key'] == 'historical':
        hist_scenario = None
    else:
        # Load historical emissions and temperature response xarray
        input_xarrays.update(historical=load_emissions_dataset(os.path.join(cfg['evaluation']['dirpath'], 'inputs_historical.nc')))
        output_xarrays.update(historical=load_response_dataset(os.path.join(cfg['evaluation']['dirpath'], 'outputs_historical.nc')))

        # Create historical scenario instances
        hist_scenario = make_scenario(key='historical', inputs=input_xarrays, outputs=output_xarrays)

    # Create test scenario
    test_scenario = make_scenario(key=cfg['evaluation']['key'], inputs=input_xarrays, outputs=output_xarrays, hist_scenario=hist_scenario)
    return test_scenario.to(device)


def predict(model, scenario, cfg):
    # Setup indices on which inference is run
    time_idx = torch.arange(cfg['evaluation']['time_idx']['start'], cfg['evaluation']['time_idx']['end'])
    lat_idx = torch.arange(cfg['evaluation']['lat_idx']['start'], cfg['evaluation']['lat_idx']['end'])
    lon_idx = torch.arange(cfg['evaluation']['lon_idx']['start'], cfg['evaluation']['lon_idx']['end'])

    # Compute prior mean
    prior_mean = model._compute_mean(scenario)[time_idx][:, lat_idx][:, :, lon_idx]

    # Extract groundtruth
    groundtruth = scenario.tas[time_idx][:, lat_idx][:, :, lon_idx]

    # Predict
    with torch.no_grad():
        qf = model(scenario, time_idx, lat_idx, lon_idx, diag=True)
        posterior_mean = qf.mean
        posterior_stddev = torch.sqrt(qf.variance)

        posterior_mean = model.mu_targets + model.sigma_targets * posterior_mean
        posterior_mean = posterior_mean.reshape(prior_mean.shape)
        posterior_mean = prior_mean + posterior_mean

        posterior_stddev = model.sigma_targets * posterior_stddev
        posterior_stddev = posterior_stddev.reshape(prior_mean.shape)

    # Encapsulate into named tuple object of xarrays
    field_names = ['prior_mean', 'posterior_mean', 'posterior_stddev', 'groundtruth']
    Pred = namedtuple(typename='Pred', field_names=field_names, defaults=(None,) * len(field_names))
    encapsulate = lambda data: encapsulate_as_xarray(data, scenario, time_idx, lat_idx, lon_idx)
    kwargs = {'prior_mean': encapsulate(prior_mean),
              'posterior_mean': encapsulate(posterior_mean),
              'posterior_stddev': encapsulate(posterior_stddev),
              'groundtruth': encapsulate(groundtruth)}
    pred = Pred(**kwargs)
    return pred


def encapsulate_as_xarray(data, scenario, time_idx, lat_idx, lon_idx):
    field = xr.DataArray(data=data.cpu(),
                         dims=['time', 'lat', 'lon'],
                         coords=dict(time=scenario.timesteps[time_idx].cpu(),
                                     lat=scenario.lat[lat_idx].cpu(),
                                     lon=scenario.lon[lon_idx].cpu()))
    return field


def compute_scores(pred):
    # Compute weighted RMSE
    rmse = metrics.weighted_rmse(pred.groundtruth, pred.posterior_mean)

    # Compute weighted NLL
    nll = metrics.weighted_nll(pred.groundtruth, pred.posterior_mean, pred.posterior_stddev)

    # Compute calibration scores
    ICI, calib95 = metrics.compute_calib95_ICI(pred.groundtruth, pred.posterior_mean, pred.posterior_stddev)

    # Encapsulate in dict
    scores = {'rmse': rmse,
              'nll': nll,
              'ICI': ICI,
              'calib95': calib95}
    return scores


def dump_plots(pred, cfg, output_dir):
    # Plot tryptychs
    for idx in cfg['evaluation']['plot']['tryptych']['indices']:
        year = int(pred.groundtruth.time.data[idx])
        dump_path = os.path.join(output_dir, f"tryptych_{year}.jpg")
        fig, ax = vis.plot_tryptych(pred.prior_mean.isel(time=idx),
                                    pred.posterior_mean.isel(time=idx) - pred.prior_mean.isel(time=idx),
                                    pred.posterior_mean.isel(time=idx),
                                    pred.posterior_mean.isel(time=idx) + 2 * pred.posterior_stddev.isel(time=idx),
                                    pred.posterior_mean.isel(time=idx) - 2 * pred.posterior_stddev.isel(time=idx),
                                    pred.groundtruth.isel(time=idx))
        plt.savefig(dump_path)
        plt.close()

    # Plot timeserie
    dump_path = os.path.join(output_dir, "timeserie.jpg")
    ts_indices = cfg['evaluation']['plot']['timeserie']['indices']
    fig, ax = vis.plot_timeserie_maps(pred.posterior_mean.isel(time=ts_indices))
    plt.savefig(dump_path)
    plt.close()


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Load model config file
    with open(os.path.join(args['--model'], 'cfg.yaml'), "r") as f:
        model_cfg = yaml.safe_load(f)

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
    main(args, cfg, model_cfg)
