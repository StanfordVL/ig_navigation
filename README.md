# Requirements

* Ubuntu (18.04 or latest LTS release)
* Up-to-date Nvidia driver
* 64 GB ram (~ 6 gb per parallel environment)

# Introduction

This repository showcases how to make an "intermediate" level project with iGibson, it includes:

* Distributed training with ray
* declarative configuration of experiments with hydra
* using a custom model override for ray capable of mixed observation modalities
* rendering the environment with a parallelized evaluation environment during training
* implementing a custom task, custom termination condition, and custom reward function
* saving/loading agent state
* pinning all dependencies, and seeding the environment, for reproducible RL

# Installation 

Deterministic installation via conda-lock:

```bash
git clone git@github.com:mjlbach/ig_navigation
conda install -c conda-forge 'conda-lock[pip_support]'
conda-lock install -n ig_navigation
conda activate ig_navigation
pip install -e .
```

Regular installation (reproducible result not guaranteed)

```bash
git clone git@github.com:mjlbach/ig_navigation
conda env create -f environment.yml
conda activate ig_navigation
pip install -e .
```

# Running

Before running, you will want to specify the path to `ig_assets` and `ig_dataset`:

```bash
export GIBSON_ASSETS_PATH=/home/michael/Documents/ig_data/assets
export IGIBSON_DATASET_PATH=/home/michael/Documents/ig_data/ig_dataset
```

This can be done automatically via a `.envrc` file and [direnv](https://direnv.net/).

To run with the default settings (experiments will be dumped to `ray_results` in the current folder:

```bash
python train.py ++experiment_save_path=$HOME/ray_results
```

This repo uses [hydra](https://hydra.cc/) to allow a mix of declarative and argument based configuration,
select the experiment with `+experiment=path`:

```bash
python train.py +experiment=search ++experiment_name=my_experiment_name ++experiment_save_path=$(pwd)/ray_results
```

Note: This will take ~64 gb of ram. If this is not feasible, you can lower the number of environments used with an additional hydra flag,
or by overriding in the experiment yaml (the default is at `configs/experiments/search.yaml`:

```bash
python train.py ++num_envs=4
```

# Reproducibility notes

Developer dependencies are locked with conda-lock
```bash
conda-lock -c pytorch -c conda-forge -p linux-64
```
