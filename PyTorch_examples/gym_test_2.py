import gym
import math
import random
import numpy as np
import matplotlib

from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()

env = gym.make('CartPole-v0').unwrapped
env.reset()

screen = env.render(mode='rgb_array')

env.close()

