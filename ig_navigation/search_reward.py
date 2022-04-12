import numpy as np
from igibson.object_states.robot_related_states import ObjectsInFOVOfRobot
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class SearchReward(BaseRewardFunction):
    """
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(SearchReward, self).__init__(config)
        self.success_reward = self.config["success_reward_scaling"]
        self.dist_tol = self.config.get("dist_tol", 1.0)
        self.name = "search_reward"

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        success = (
            l2_distance(
                env.robots[0].get_position()[:2], task.target_obj.get_position()[:2]
            )
            < self.dist_tol
        )

        body_ids = env.robots[0].states[ObjectsInFOVOfRobot].get_value()
        in_view = task.target_obj.get_body_ids()[0] in body_ids
        success = success and in_view
        reward = self.success_reward if success else 0.0
        return reward


class PotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super(PotentialReward, self).__init__(config)
        self.potential_reward_weight = self.config["potential_reward_scaling"]
        self.name = "potential_reward"

    def reset(self, task, env):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.potential = self.get_shortest_path(env, task)

    def get_shortest_path(self, env, task, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        source = env.robots[0].get_position()[:2]
        target = task.target_obj.get_position()[:2]
        _, geodesic_distance = np.array(
            env.scene.get_shortest_path(0, source, target, entire_path=entire_path)
        )
        return geodesic_distance

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        new_potential = self.get_shortest_path(env, task)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential
        return reward
