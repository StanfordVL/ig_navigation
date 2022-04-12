from ray.rllib.agents.callbacks import DefaultCallbacks
import cv2
from pathlib import Path


class MetricsCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        reward_breakdown = episode.last_info_for()["reward_breakdown"]
        if "reward_breakdown" not in episode.user_data:
            episode.user_data["reward_breakdown"] = {}
            for key, value in reward_breakdown.items():
                episode.user_data["reward_breakdown"][key] = value
        else:
            for key, value in reward_breakdown.items():
                episode.user_data["reward_breakdown"][key] += value

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        # Make sure this episode is really done.
        assert episode.batch_builder.policy_collectors["default_policy"].batches[-1][
            "dones"
        ][-1], (
            "ERROR: `on_episode_end()` should only be called " "after episode is done!"
        )
        reward_breakdown = episode.user_data["reward_breakdown"]
        for key, value in reward_breakdown.items():
            if key == "total":
                continue
            episode.custom_metrics[key] = value

        for key, value in reward_breakdown.items():
            if key == "total":
                continue
            if reward_breakdown["total"] == 0:
                episode.custom_metrics[key + "_fraction"] = 0
            else:
                episode.custom_metrics[key + "_fraction"] = (
                    value / reward_breakdown["total"]
                )
        episode.custom_metrics["success"] = int(episode.last_info_for()["success"])


class DummyCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        pass
