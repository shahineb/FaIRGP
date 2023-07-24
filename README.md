# FaIRGP: A Bayesian Energy Balance Model for Surface Temperatures Emulation


<p align="center">
  <img width="40%" src="docs/img/fairgp-logo.png"/>
</p>


# Getting started

## Running comparison with baselines for global SSP emulation

#### Running evaluation of FaIRGP
- Run from root directory
```bash
$ python evaluate_FaIRGP.py --cfg=config/FaIRGP.yaml --o=path/to/output/directory
```

#### Running evaluation of Plain GP
- Run from root directory
```bash
$ python evaluate_Plain_GP.py --cfg=config/PlainGP.yaml --o=path/to/output/directory
```

#### Running evaluation of FaIR
- Run from root directory
```bash
$ python evaluate_FaIR.py --cfg=config/FaIR.yaml --o=path/to/output/directory
```



## Reproducing paper results

#### *Simulation Example*

- Run experiment with multiple initialisation seeds
```bash
$ source ./repro/repro_mvn_experiment_multi_seeds.sh
```

- Run ablation study on number of training samples
```bash
$ source ./repro/repro_mvn_experiment_ntrain.sh
```

- Run ablation study on number semi-supervised samples
```bash
$ source ./repro/repro_mvn_experiment_semiprop.sh
```

- Run ablation study on number of dimensionality of X2
```bash
$ source ./repro/repro_mvn_experiment_d_X2.sh
```

- Run experiment for random forest model
> Go to `notebooks/mvn-random-forest-models.ipynb`


- Visualise scores and generate plots
> Go to `notebooks/mvn-experiments-score-analysis.ipynb`


#### *Aerosol Radiative Forcing Example*

- Run experiment with multiple initialisation seeds
```bash
$ source ./repro/repro_FaIR_experiment_multi_seeds.sh
```

- Run experiment for random forest model
> Go to `notebooks/FaIR-random-forest-models.ipynb`


- Visualise scores and generate table
> Go to `notebooks/FaIR-experiments-score-analysis.ipynb`






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
