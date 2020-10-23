import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


#
# # define policy network
# class cnn_policy_net(nn.Module):
#     def __init__(
#         self, nS, nH, nA
#     ):  # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
#         super(policy_net, self).__init__()
#         self.h = nn.Linear(nS, nH)
#         self.out = nn.Linear(nH, nA)
#
#     # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
#     def forward(self, x):
#         x = F.relu(self.h(x))
#         x = F.softmax(self.out(x), dim=1)
#         return x
#
#
# class Print(nn.Module):
#     def __init__(self):
#         super(Print, self).__init__()
#
#     def forward(self, x):
#         print("layer input:", x.shape)
#         return x


class cnn_policy_net(nn.Module):
    def __init__(self, num_inputs, hidden_size=128):
        super(cnn_policy_net, self).__init__()

        # For 80x60 input
        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Print(),
            Flatten(),
            nn.Linear(32 * 7 * 5, hidden_size),
            nn.ReLU(),
        )

        # init_ = lambda m: init(m,
        #     nn.init.orthogonal_,
        #     lambda x: nn.init.constant_(x, 0))

        # self.critic_linear = init_(nn.Linear(hidden_size, 1))

        # self.train()

    def forward(self, x):
        x = self.main(x)

        return x
