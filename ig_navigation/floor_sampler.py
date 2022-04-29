import numpy as np
import pybullet as p
from igibson.external.pybullet_tools.utils import (get_aabb_center,
                                                   get_aabb_extent,
                                                   stable_z_on_aabb)
from igibson.object_states.aabb import AABB
from igibson.object_states.object_state_base import CachingEnabledObjectState
from igibson.utils.utils import l2_distance, restoreState


def get_center_extent(obj_states):
    assert AABB in obj_states
    aabb = obj_states[AABB].get_value()
    center, extent = get_aabb_center(aabb), get_aabb_extent(aabb)
    return center, extent


def clear_cached_states(obj):
    for _, obj_state in obj.states.items():
        if isinstance(obj_state, CachingEnabledObjectState):
            obj_state.clear_cached_value()


def detect_collision(bodyA):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def sample_on_floor(
    objA,
    scene,
    room,
    max_trials=1000,
    skip_falling=False,
):

    objA.force_wakeup()

    success = False
    state_id = p.saveState()

    pos, orientation = objA.get_position_orientation()

    for _ in range(max_trials):
        pos = None
        if hasattr(objA, "orientations") and objA.orientations is not None:
            orientation = objA.sample_orientation()
        else:
            orientation = [0, 0, 0, 1]

        # Orientation needs to be set for stable_z_on_aabb to work correctly
        # Position needs to be set to be very far away because the object's
        # original position might be blocking rays (use_ray_casting_method=True)
        old_pos = np.array([200, 200, 200])
        objA.set_position_orientation(old_pos, orientation)

        _, pos = scene.get_random_point_by_room_instance(room)
        # pos = np.array([0.0, 0.0, -1.0])

        pos[2] = stable_z_on_aabb(
            objA.get_body_ids()[0], ([0, 0, pos[2]], [0, 0, pos[2]])
        )
        pos[2] += 0.3
        objA.set_position_orientation(pos, orientation)
        p.stepSimulation()
        success = not detect_collision(
            objA.get_body_ids()[0]
        )  # len(p.getContactPoints(objA.get_body_id())) == 0
        break
        if success:
            break
        else:
            restoreState(state_id)

    p.removeState(state_id)

    if success and not skip_falling:
        objA.set_position_orientation(pos, orientation)
        # Let it fall for 0.2 second
        physics_timestep = p.getPhysicsEngineParameters()["fixedTimeStep"]
        for _ in range(int(0.2 / physics_timestep)):
            p.stepSimulation()
            if len(p.getContactPoints(bodyA=objA.get_body_ids()[0])) > 0:
                break

    return success

def sample_initial_pose_and_target_pos(env):
    """
    Sample robot initial pose and target position

    :param env: environment instance
    :return: initial pose and target position
    """
    _, initial_pos = env.scene.get_random_point(floor=0)
    max_trials = 100
    dist = 0.0
    for _ in range(max_trials):
        _, target_pos = env.scene.get_random_point(floor=0)
        if env.scene.build_graph:
            _, dist = env.scene.get_shortest_path(
                0, initial_pos[:2], target_pos[:2], entire_path=False
            )
        else:
            dist = l2_distance(initial_pos, target_pos)
        if 1.0 < dist < 10.0:
            break
    # if not (self.target_dist_min < dist < self.target_dist_max):
        # log.warning("Failed to sample initial and target positions")
    initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
    # log.debug("Sampled initial pose: {}, {}".format(initial_pos, initial_orn))
    # log.debug("Sampled target position: {}".format(target_pos))
    return initial_pos, initial_orn, target_pos

def reset_agent(env):
    """
    Reset robot initial pose.
    Sample initial pose and target position, check validity, and land it.

    :param env: environment instance
    """
    reset_success = False
    max_trials = 100

    # cache pybullet state
    # TODO: p.saveState takes a few seconds, need to speed up
    state_id = p.saveState()
    for i in range(max_trials):
        # initial_pos, initial_orn, target_pos = sample_initial_pose_and_target_pos(env)
        _, initial_pos = env.scene.get_random_point(floor=0)
        initial_orn = [0,0,0]
        reset_success = env.test_valid_position(
            env.robots[0], initial_pos, initial_orn
        )# and env.test_valid_position(env.robots[0], target_pos)
        restoreState(state_id)
        if reset_success:
            break

    if not reset_success:
        print("WARNING: Failed to reset robot without collision")
        # TODO: setting the variables to None so that any code relying on these values fails
        # It is maybe better to create an assert here as we do not want to continue if we couldn't sample
        target_pos = None
        initial_pos = None
        initial_orn = None

    target_pos = [0,0,0]
    p.removeState(state_id)
    return initial_pos, initial_orn, target_pos
