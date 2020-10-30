import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print("layer input:", x.shape)
        return x


class cnn_policy_net(nn.Module):
    def __init__(self, n_channels, n_actions):
        super(cnn_policy_net, self).__init__()

        # For 128x128 input
        self.main_chunk = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Print(),
            Flatten(),
        )

        self.remaining = nn.Sequential(nn.Linear(hidden_size, n_actions), nn.Softmax(dim=-1),)

    def forward(self, obs):
        return self.remaining(self.main_chunk(obs))

    def act(self, obs):
        probs = self.remaining(self.main_chunk(obs))
        dist = Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate(self, obs, action):
        probs = self.remaining(self.main_chunk(obs))
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return entropy, log_prob


class cnn_value_net(nn.Module):
    def __init__(self, n_channels, n_actions):
        super(cnn_value_net, self).__init__()

        # For 128x128 input
        self.main_chunk = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Print(),
            Flatten(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs):
        return self.main_chunk(obs)
