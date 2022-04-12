import os

import numpy as np
from bddl.object_taxonomy import ObjectTaxonomy
from igibson.object_states.robot_related_states import ObjectsInFOVOfRobot
from igibson.objects.articulated_object import URDFObject
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.termination_condition_base import \
    BaseTerminationCondition
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.assets_utils import (get_ig_avg_category_specs,
                                        get_ig_category_path,
                                        get_ig_model_path)
from igibson.utils.utils import l2_distance

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
                env.robots[0].get_position()[:2], task.target_obj.get_position()[:2]
            )
            < self.dist_tol
        )
        in_view = (
            task.target_obj.main_body
            in env.robots[0].states[ObjectsInFOVOfRobot].get_value()
        )
        done = done and in_view
        success = done
        return done, success


class SearchTask(BaseTask):
    def __init__(self, env):
        super(SearchTask, self).__init__(env)
        self.object_taxonomy = ObjectTaxonomy()
        self.avg_category_spec = get_ig_avg_category_specs()

        self.scene = env.scene
        self.simulator = env.simulator
        self.sampleable_categories = self.get_sampleable_categories()

        self.reward_functions = []
        self.reward_functions.append(SearchReward(self.config))
        self.reward_functions.append(PotentialReward(self.config))

        self.termination_conditions = [
            Timeout(self.config),
            SearchTermination(self.config),
        ]
        self.choose_task()
        self.task_obs_dim = 3

    def get_sampleable_categories(self):
        leaf_nodes = []
        for node in self.object_taxonomy.taxonomy.nodes:
            if self.object_taxonomy.taxonomy.out_degree(node) == 0:
                if self.object_taxonomy.taxonomy.in_degree(node) == 1:
                    if node not in ["floor.n.01"]:
                        leaf_nodes.append(node)
        return leaf_nodes

    def choose_task(self):
        # obj_pro = self.import_object(wordnet_category = 'microwave.n.02' , model='7128')
        obj_pro = self.import_object(igibson_category="microwave", model="7320")
        self.target_obj = obj_pro
        room = np.random.choice(np.array(list(self.scene.room_ins_name_to_ins_id)))
        sample_on_floor(obj_pro, self.scene, room=room)

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
        env.robots[0].reset()
        if self.config.get("randomize_agent_reset", False):
            room = np.random.choice(
                np.array(list(self.scene.room_ins_name_to_ins_id.keys()))
            )
            sample_on_floor(env.robots[0], env.simulator.scene, room)
        else:
            env.land(env.robots[0], [1.0, 2.0, 0.0], [0, 0, 0])

    def reset_scene(self, env):
        # This is absolutely critical, reset doors
        env.scene.reset_scene_objects()

        if self.config.get("randomize_obj_reset", True):
            room = np.random.choice(
                np.array(list(self.scene.room_ins_name_to_ins_id.keys()))
            )
            sample_on_floor(self.target_obj, self.scene, room=room)
        else:
            self.target_obj.set_position(
                np.array([0.79999999, -3.19999984, 0.48399325])
            )

    def import_object(
        self,
        wordnet_category=None,
        igibson_category=None,
        model=None,
        scale=None,
        use_bbox=True,
    ):
        if wordnet_category:
            categories = self.object_taxonomy.get_subtree_igibson_categories(
                wordnet_category
            )
            igibson_category = np.random.choice(categories)
        else:
            assert igibson_category is not None
        if model:
            pass
        else:
            category_path = get_ig_category_path(igibson_category)
            model_choices = os.listdir(category_path)

            # Randomly select an object model
            model = np.random.choice(model_choices)

        model_path = get_ig_model_path(igibson_category, model)
        filename = os.path.join(model_path, model + ".urdf")
        num_new_obj = len(self.scene.objects_by_name)
        obj_name = "{}_{}".format(igibson_category, num_new_obj)
        # create the object and set the initial position to be far-away
        simulator_obj = URDFObject(
            filename,
            name=obj_name,
            category=igibson_category,
            scale=scale,
            model_path=model_path,
            avg_obj_dims=self.avg_category_spec.get(igibson_category),
            fit_avg_dim_volume=use_bbox,
            texture_randomization=False,
            overwrite_inertial=True,
        )

        # # Load the object into the simulator
        self.simulator.import_object(simulator_obj)
        # task
        return simulator_obj
