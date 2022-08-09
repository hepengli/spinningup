"""Core functions of the GPO algorithm."""
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def forward(self, obs, n=None):
        return NotImplementedError

    def kl(self, other, obs):
        raise NotImplementedError

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def forward(self, obs, n=None):
        return self._distribution(obs).sample()

    def kl(self, other, obs):
        self_logits = self.logits_net(obs)
        other_logits = other.logits_net(obs)
        a0 = self_logits - self_logits.max(dim=-1, keepdims=True).values
        a1 = other_logits - other_logits.max(dim=-1, keepdims=True).values
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = ea0.sum(dim=-1, keepdims=True)
        z1 = ea1.sum(dim=-1, keepdims=True)
        p0 = ea0 / z0
        kl = p0 * (a0 - torch.log(z0) - a1 + torch.log(z1))
        return kl.sum(dim=-1)

    def prob(self, obs):
        logits = self.logits_net(obs)
        return torch.nn.Softmax(logits)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_space, hidden_sizes, activation):
        super().__init__()
        act_dim = act_space.shape[0]
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self._low = torch.as_tensor(act_space.low, dtype=torch.float32)
        self._high = torch.as_tensor(act_space.high, dtype=torch.float32)

    def forward(self, obs, test=False, n=1):
        mu = self.mu_net(obs)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        value = mu if test else dist.rsample((n,))
        action = self._low + (self._high - self._low) * (torch.tanh(value) + 1) / 2

        return action

    def kl(self, other, obs):
        mu1, mu2 = self.mu_net(obs), other.mu_net(obs)
        log_std1 = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std2 = torch.clamp(other.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std1 = torch.exp(log_std1)
        std2 = torch.exp(log_std2)
        kl = log_std2 - log_std1 + (torch.square(std1) + \
            torch.square(mu1 - mu2)) / (2.0 * torch.square(std2)) - 0.5

        return kl.sum(dim=-1)


class MLPContinuousCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPDeiscreteCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs, act):
        return self.q(obs).select(-1, act)


class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy anc critic builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space, hidden_sizes, activation)
            self.q1  = MLPContinuousCritic(obs_dim, action_space.shape[0], hidden_sizes, activation)
            self.q2  = MLPContinuousCritic(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
            self.q1  = MLPContinuousCritic(obs_dim, action_space.n, hidden_sizes, activation)
            self.q2  = MLPContinuousCritic(obs_dim, action_space.n, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            return self.pi(obs, test=deterministic).numpy()
