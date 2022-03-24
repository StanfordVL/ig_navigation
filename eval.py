import re
import cv2
import hydra
from omegaconf import OmegaConf
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from pathlib import Path

from ig_navigation.callbacks import MetricsCallback


def igibson_env_creator(env_config):
    from ig_navigation.igibson_env import SearchEnv

    return SearchEnv(
        config_file=env_config,
        mode=env_config["mode"],
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )

@hydra.main(config_path=ssg.CONFIG_PATH, config_name="config")
def main(cfg):
    ray.init()
    env_config = OmegaConf.to_object(cfg)
    register_env("igibson_env_creator", igibson_env_creator)
    checkpoint_path = Path(cfg.experiment_save_path, cfg.experiment_name)
    config = {
        "env": "igibson_env_creator",
        "model": OmegaConf.to_object(cfg.model),
        "env_config": env_config,  # config to pass to env class
        "num_workers": 0,
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

    agent = ppo.PPOTrainer(
        config,
    )

    if Path(checkpoint_path).exists():
        checkpoints = Path(checkpoint_path).rglob("checkpoint-*")
        checkpoints = [
            str(f) for f in checkpoints if re.search(r".*checkpoint-\d*$", str(f))
        ]
        checkpoints = sorted(checkpoints)
        if len(checkpoints) > 0:
            agent.restore(checkpoints[-1])

    env = igibson_env_creator(env_config)
    env.reset()
    frames = []
    # run until episode ends
    trials = 0
    successes = 0
    frames.append(env.render())

    video_folder = Path('eval', cfg.experiment_name, 'videos')
    video_folder.mkdir(parents = True, exist_ok = True)
    video_path = f'eval_episodes.mp4'
    video = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        frames[0].shape[:2]
    )

    for _ in range(100):
        episode_reward = 0
        done = False
        obs = env.reset()
        success = False
        reward_breakdown = defaultdict(lambda: 0)
        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            frames.append(env.render())
            for reward, value in info['reward_breakdown'].items():
                reward_breakdown[reward] += value
            if info['success']:
                assert done
                success = True
                successes +=1
        trials += 1
        print('Success: ', success)
        print('episode reward: ', episode_reward)
        for key, value in reward_breakdown.items():
            print(f"{key}: {value}")
        print()

    print(f"success fraction {successes/trials}")
    for frame in frames:
        screen = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(screen)
    video.release()

if __name__ == "__main__":
    main()
