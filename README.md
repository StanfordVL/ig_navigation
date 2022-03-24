# Installation 
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
python train.py +experiment=search ++experiment_name=my_experiment_name ++experiment_save_path=$(pwd)/ray_results
```

# Updating
```bash
conda env update --file environment.yml --prune
```
