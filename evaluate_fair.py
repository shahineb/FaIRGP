"""
Description : Runs and evaluates FaIR for global temperature response emulation

Usage: evaluate_fair.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Device to use [default: cpu]
"""
import os
import yaml
import logging
from docopt import docopt
import pandas as pd
import torch
from src.preprocessing import make_data
from src.evaluation.metrics import compute_deterministic_metrics
import src.fair as fair


def main(args, cfg):
    # Create dataset
    logging.info("Loading dataset")
    data = make_data(cfg=cfg)

    # Initialize empty list for scores
    scores = []

    # Run FaIR on different scenarios
    logging.info("\n Running FaIR")
    for scenario in data.scenarios.scenarios.values():
        if scenario.name == 'historical':
            continue
        else:
            fair_tas = run_fair(scenario=scenario, cfg=cfg)
            scenario_score = evaluate(fair_tas=fair_tas, groundtruth_tas=scenario.tas)
            scores.append(scenario_score)

    # Dump scores
    dump_scores(scores=scores, output_dir=args['--o'])


def run_fair(scenario, cfg):
    base_kwargs = fair.get_params()
    res = fair.run(time=scenario.full_timesteps.numpy(),
                   emission=scenario.full_emissions.T.numpy(),
                   base_kwargs=base_kwargs)
    tas = scenario.trim_hist(res['T'])
    return tas


def evaluate(fair_tas, groundtruth_tas):
    # Compute scores
    scores = compute_deterministic_metrics(torch.from_numpy(fair_tas).float(),
                                           groundtruth_tas)
    return scores


def dump_scores(scores, output_dir):
    scores_df = pd.DataFrame(data=scores)
    scores_df.to_csv(os.path.join(output_dir, 'cv-scores.csv'), index=False)


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
