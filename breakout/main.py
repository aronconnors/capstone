import gym
import numpy as np
from PIL import Image
import torch
from breakout import *
import os
from model import AtariNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device, render_mode='human')

model = AtariNet(nb_actions=4)

model.to(device)

model.load_the_model()

state = environment.reset()

print(model.forward(state))

"""
for _ in range(100):
    action = environment.action_space.sample()

    state, reward, done, info = environment.step(action)

    print(state.shape)"""