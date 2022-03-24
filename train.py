import re
import numpy as np
import hydra
from omegaconf import OmegaConf
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from pathlib import Path
from ray.tune.logger import UnifiedLogger

import ig_navigation
from ig_navigation.callbacks import MetricsCallback, DummyCallback


def igibson_env_creator(env_config):
    from ig_navigation.igibson_env import SearchEnv

    return SearchEnv(
        config_file=env_config,
        mode=env_config["mode"],
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )


@hydra.main(config_path=ig_navigation.CONFIG_PATH, config_name="config")
def main(cfg):
    ray.init()
    env_config = OmegaConf.to_object(cfg)
    register_env("igibson_env_creator", igibson_env_creator)
    checkpoint_path = Path(cfg.experiment_save_path, cfg.experiment_name)

    num_epochs = np.round(cfg.training_timesteps / cfg.n_steps).astype(int)
    save_ep_freq = np.round(
        num_epochs / (cfg.training_timesteps / cfg.save_freq)
    ).astype(int)

    config = {
        "env": "igibson_env_creator",
        "model": OmegaConf.to_object(cfg.model),
        "env_config": env_config,  # config to pass to env class
        "num_workers": cfg.num_envs,
        "framework": "torch",
        "seed": cfg.seed,
        "lambda": cfg.gae_lambda,
        "lr": cfg.learning_rate,
        "train_batch_size": cfg.n_steps,
        "rollout_fragment_length": cfg.n_steps // cfg.num_envs,
        "num_sgd_iter": cfg.n_epochs,
        "sgd_minibatch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "create_env_on_driver": False,
        "num_gpus": 1,
        "callbacks": MetricsCallback,
        # "log_level": "DEBUG",
        # "_disable_preprocessor_api": False,
    }

    if cfg.eval_freq > 0 and not cfg.debug:
        eval_ep_freq = np.round(
            num_epochs / (cfg.training_timesteps / cfg.eval_freq)
        ).astype(int)
        config.update(
            {
                "evaluation_interval": eval_ep_freq,  # every n episodes evaluation episode
                "evaluation_duration": 20,
                "evaluation_duration_unit": "episodes",
                "evaluation_num_workers": 1,
                "evaluation_parallel_to_training": True,
                "evaluation_config": {
                    "callbacks": DummyCallback,
                    "record_env": True,
                },
            }
        )

    log_path = str(checkpoint_path.joinpath("log"))
    Path(log_path).mkdir(parents=True, exist_ok=True)
    trainer = ppo.PPOTrainer(
        config,
        logger_creator=lambda x: UnifiedLogger(x, log_path),
    )

    if Path(checkpoint_path).exists():
        checkpoints = Path(checkpoint_path).rglob("checkpoint-*")
        checkpoints = [
            str(f) for f in checkpoints if re.search(r".*checkpoint-\d*$", str(f))
        ]
        checkpoints = sorted(checkpoints)
        if len(checkpoints) > 0:
            trainer.restore(checkpoints[-1])

    for i in range(num_epochs):
        # Perform one iteration of training the policy with PPO
        trainer.train()

        if (i % save_ep_freq) == 0:
            checkpoint = trainer.save(checkpoint_path)
            print("checkpoint saved at", checkpoint)


if __name__ == "__main__":
    main()
