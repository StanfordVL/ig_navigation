import os
from ig_navigation.turtlebot import Turtlebot
from igibson.robots.robot_base import REGISTERED_ROBOTS

ROOT_PATH= os.path.dirname(__file__)
CONFIG_PATH=os.path.join(ROOT_PATH, '..', 'configs')

REGISTERED_ROBOTS["Turtlefast"] = Turtlebot
