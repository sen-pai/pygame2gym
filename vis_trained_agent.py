import gym
import simple_discrete_game
import matplotlib.pyplot as plt
from array2gif import write_gif
import torch
import torch.nn as nn
import numpy as np

from models.mlp_agent import mlp_policy_net, mlp_value_net



def preprocess_obs(obs):
    # convert to grayscale by averaging channels
    obs = np.mean(obs, axis=2) / 255.0
    # convert to tensor
    return torch.from_numpy(obs).float().view(-1)



env = gym.make("GoalGrid-v0")
state_size = 128 * 128
n_actions = env.action_space.n

actor = mlp_policy_net(state_size, 32, n_actions)
actor.load_state_dict(torch.load("ppo_GoalGrid-v0_actor30.pth"))
all_states = []
episodes = 50
for i in range(10):
    obs = env.reset()
    for e in range(episodes):
        obs= preprocess_obs(obs)
        action, _ = actor.act(obs)
        obs, _, done, _ = env.step(action.item())
        all_states.append(obs)
        if done:
            break
    write_gif(all_states, str(i) + ".gif", fps=10)
    all_states = []
