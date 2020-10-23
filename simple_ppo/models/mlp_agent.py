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


# define discrete policy network
class mlp_policy_net(nn.Module):
    def __init__(self, state_size, hidden_size, n_actions):
        super(mlp_policy_net, self).__init__()

        self.main_chunk = nn.Sequential(
            nn.Linear(state_size, hidden_size * 2), Mish(), nn.Linear(hidden_size * 2, hidden_size),
        )

        self.remaining = nn.Sequential(
            Mish(), nn.Linear(hidden_size, n_actions), nn.Softmax(dim=-1),
        )

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

    def pi_representation(self, obs):
        return self.main_chunk(obs)


# define value network
class mlp_value_net(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(mlp_value_net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size, hidden_size * 2),
            Mish(),
            nn.Linear(hidden_size * 2, hidden_size),
            Mish(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs):
        return self.main(obs)


class discriminator_mlp_net(nn.Module):
    def __init__(self, state_size, actions_size, hidden_size):
        super(discriminator_mlp_net, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(state_size + actions_size, hidden_size * 2),
            Mish(),
            nn.Linear(hidden_size * 2, hidden_size),
            Mish(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs, action):
        s_a = torch.cat((obs, action), 1)
        return self.main(s_a)


# define discrete policy network
class not_mlp_policy_net(nn.Module):
    def __init__(self, state_size, other_policy_representation, hidden_size, n_actions):
        super(not_mlp_policy_net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size + other_policy_representation, hidden_size * 2),
            Mish(),
            nn.Linear(hidden_size * 2, hidden_size),
            Mish(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, obs):
        return self.main(obs)

    def act(self, obs):
        probs = self.main(obs)
        dist = Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate(self, obs, action):
        probs = self.main(obs)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return entropy, log_prob
