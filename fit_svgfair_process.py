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
import torch
from gpytorch import kernels, likelihoods, mlls
from src.preprocessing import make_data
from src.models import ThermalBoxesSVGP
import src.models.utils_svgp as utils
from src.evaluation import dump_state_dict
import notebooks.utils_spatial as spatial
