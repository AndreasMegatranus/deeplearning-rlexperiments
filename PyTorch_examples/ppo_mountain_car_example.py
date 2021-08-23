import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter


# This is based on 
# https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char07%20PPO/PPO_MountainCar-v0.py
# https://awesomeopensource.com/project/sweetice/Deep-reinforcement-learning-with-pytorch


# Parameters
env_name = 'MountainCar-v0'
gamma = 0.99
render = False
seed = 1
log_interval = 10


# Set up env

env = gym.make(env_name).unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

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


# Actor and Critic definitions
# Both are simple 2-layer NNs.

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

# The use of softmax here is consistent with a computed action that is discrete rather than continuous.


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value
