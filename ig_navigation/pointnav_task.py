import os

import numpy as np
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.termination_condition_base import \
    BaseTerminationCondition
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.utils import l2_distance

from ig_navigation import floor_sampler
from ig_navigation.floor_sampler import sample_on_floor
from ig_navigation.search_reward import PotentialReward, SearchReward


class SearchTermination(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(SearchTermination, self).__init__(config)
        self.dist_tol = self.config.get("dist_tol", 1.0)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = (
            l2_distance(
                env.robots[0].get_position()[:2], task.goal
            )
            < self.dist_tol
        )
        success = done
        return done, success


class PointNavTask(BaseTask):
    def __init__(self, env):
        super(PointNavTask, self).__init__(env)
        self.scene = env.scene
        self.simulator = env.simulator
        
        self.reward_functions = []
        self.reward_functions.append(SearchReward(self.config))
        self.reward_functions.append(PotentialReward(self.config))
        self.is_interactive = self.config["scene"] == "igibson"

        self.termination_conditions = [
            Timeout(self.config),
            SearchTermination(self.config),
        ]
        # TODO: get from config if the goal is changing or if it is constant, and the value of it if it is constant
        self.goal = np.array([0.79999999, -3.19999984, 0.48399325])

    def get_reward(self, env, _collision_links=[], _action=None, info={}):
        """
        Aggreate reward functions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        reward = 0.0
        categorical_reward_info = {}
        for reward_function in self.reward_functions:
            category_reward = reward_function.get_reward(self, env)
            reward += category_reward
            categorical_reward_info[reward_function.name] = category_reward

        categorical_reward_info["total"] = reward
        info["reward_breakdown"] = categorical_reward_info

        return reward, info

    def reset_agent(self, env):
        if self.is_interactive and self.config.get("randomize_agent_reset", False):
            env.robots[0].reset()
            room = np.random.choice(
                np.array(list(self.scene.room_ins_name_to_ins_id.keys()))
            )
            sample_on_floor(env.robots[0], env.simulator.scene, room)
        else:
            env.robots[0].reset()
            initial_pos, initial_orn, target_pos = floor_sampler.reset_agent(env)
            env.robots[0].set_position_orientation(initial_pos, [0,0,0,1])

    def reset_scene(self, env):
        # This is absolutely critical, reset doors
        if self.is_interactive:
            env.scene.reset_scene_objects()

        #TODO: if the config says we randomize the goal, we need to set a new one here
