import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch




# This outputs action and observation space information about Open Gym environments.
# (uncomment one of the env_name = lines)


# Environments (see also gym.openai.com/envs)

# Classic control (discrete action space)

# env_name = 'Acrobot-v1'
# env_name = 'CartPole-v1'
# env_name = 'MountainCar-v0'

# Classic control (continuous action space)

# env_name = 'MountainCarContinuous-v0'   # One continuous action variable
# env_name = 'Pendulum-v0'   # One continuous action variable

# Box2D
# For these, need to do
# pip install box2d-py

# env_name = 'BipedalWalker-v3'   # 4-d continuous action space
# env_name = 'LunarLander-v2'   # 4-d discrete action space
# env_name = 'BipedalWalkerHardcore-v3'   # 4-d continuous action space
# env_name = 'CarRacing-v0'   # 3-d continuous action space
env_name = 'LunarLanderContinuous-v2'   # 2-d continuous action space




seed = 1


# Set up env

env = gym.make(env_name).unwrapped
num_state = env.observation_space.shape[0]
# num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)


print('Environment action space:')
print(env.action_space)

'''
This printout shows that the MountainCar-v0 environment has a discrete action space, with 3 possible values
(presumably minus, plus, and neutral).  Note that there is a separate environment, MountainCarContinuous-v0,
which has a continuous action space.  See also
https://gym.openai.com/docs/#environments
https://gym.openai.com/envs/#classic_control
'''


print('Environment observation space:')
print(env.observation_space)
print('Environment observation space high, low:')
print(env.observation_space.high)
print(env.observation_space.low)

'''
This printout shows that the MountaintCar-v0 environment has a 2-dimensional box observation space
(presumably, one variable is position, and the other is acceleration or force).  
The first variable ranges from -1.2 to 0.6.  
The second variable ranges from -0.07 to 0.07.
'''

