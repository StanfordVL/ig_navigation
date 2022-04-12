# Requirements

* Ubuntu (18.04 or latest LTS release)
* Up-to-date Nvidia driver
* 64 GB ram (~ 6 gb per parallel environment)

# Installation 

Deterministic installation via conda-lock:

```bash
git clone git@github.com:mjlbach/ig_navigation
conda install -c conda-forge 'conda-lock[pip_support]'
conda-lock install -n ig_navigation
conda activate ig_navigation
pip install -e .
```

Regular installation (reproudcible result not guaranteed)

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

Uses hydra, select the experiment with +experiment=path

```bash
python train.py +experiment=navigation ++experiment_name=my_experiment_name ++experiment_save_path=$(pwd)/ray_results
```

# Reproducibility notes

Developer dependencies are locked with conda-lock
```bash
conda-lock -c pytorch -c conda-forge -p linux-64
```
