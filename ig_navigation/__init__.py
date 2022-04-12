import os

from igibson.robots.robot_base import REGISTERED_ROBOTS

from ig_navigation.turtlebot import Turtlebot

ROOT_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT_PATH, "..", "configs")

REGISTERED_ROBOTS["Turtlefast"] = Turtlebot
