# FaIRGP: A Bayesian Energy Balance Model for Surface Temperatures Emulation


<p align="center">
  <img width="35%" src="docs/img/fairgp-logo.png"/>
</p>


# Getting started

The data used in experiments can be obtained [here](https://zenodo.org/record/7064308) (files `train_val.tar.gz`, `test.tar.gz`) and should be placed in `./data` directory.


## Fit global FaIRGP model
1. Choose in `config/FaIRGP.yaml` which scenarios to use for training
2. Run from root directory
```bash
$ python fit_FaIRGP.py --cfg=config/FaIRGP.yaml --o=path/to/output/directory
```

## Fit spatial FaIRGP model
1. Choose in `config/PlainGP.yaml` which scenarios to use for training
2. Run from root directory
```bash
$ python fit_spatial_FaIRGP.py --cfg=config/spatial-FaIRGP.yaml --o=path/to/output/directory
```


# Reproduce paper results

## SSP global emulation benchmark

1. Running evaluation of FaIRGP
```bash
$ python evaluate_FaIRGP.py --cfg=config/FaIRGP.yaml --o=path/to/output/directory
```

2. Running evaluation of Plain GP
```bash
$ python evaluate_Plain_GP.py --cfg=config/PlainGP.yaml --o=path/to/output/directory
```

3. Running evaluation of FaIR
```bash
$ python evaluate_FaIR.py --cfg=config/FaIR.yaml --o=path/to/output/directory
```

4. Go to `notebooks/SSP-global-experiment-score-analysis.ipynb`


## SSP spatial emulation benchmark

1. Fit 4 spatial FaIRGP model on training set without {ssp126, ssp245, ssp370, ssp585}
2. Fit 4 spatial PlainGP model on training set without {ssp126, ssp245, ssp370, ssp585}
3. Go to `notebooks/SSP-spatial-experiment-score-analysis.ipynb`





## Installation

Code implemented in Python 3.8.0

#### Setting up environment

Create and activate environment (with [pyenv](https://www.devopsroles.com/install-pyenv/) here)
```bash
$ pyenv virtualenv 3.8.0 venv
$ pyenv activate venv
$ (venv)
```

Install dependencies
```bash
$ (venv) pip install -r requirements.txt
```

#### References
```
@article{bouabid2023fairgp,
  title={{FaIRGP: A Bayesian Energy Balance Model for Surface Temperatures Emulation}},
  author={Bouabid, Shahine and Sejdinovic, Dino and Watson-Parris, Duncan},
  journal={arXiv preprint arXiv:2307.10052},
  year={2023}
}
```
