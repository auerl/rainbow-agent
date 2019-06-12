#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""models.py: Network models module, containing different (reference) NN archtectures
              implemented using the PyTorch library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 
import math

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QNetwork(nn.Module):
    """Actor (Policy) Model.
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """A standard Q network with one hidden layer.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random number generator seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """A dueling network implementation

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random number generator seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.feature = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

    def forward(self, state):
        x = self.feature(state)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage - advantage.mean()


class NoisyQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(NoisyQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = NoisyLinear(fc1_units, fc2_units)
        self.fc3 = NoisyLinear(fc2_units, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state):
        state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        return action

    def reset_noise(self):
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class NoisyDuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """A dueling network implementation

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random number generator seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(NoisyDuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc_feature_1 = NoisyLinear(state_size, fc1_units)
        self.fc_advantage_1 = NoisyLinear(fc1_units, fc2_units)
        self.fc_advantage_2 = NoisyLinear(fc2_units, action_size)
        self.fc_value_1 = NoisyLinear(fc1_units, fc2_units)
        self.fc_value_2 = NoisyLinear(fc2_units, 1)

        self.feature = nn.Sequential(
            self.fc_feature_1,
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            self.fc_advantage_1,
            nn.ReLU(),
            self.fc_advantage_2
        )
        self.value = nn.Sequential(
            self.fc_value_1,
            nn.ReLU(),
            self.fc_value_2
        )

    def forward(self, state):
        x = self.feature(state)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage - advantage.mean()

    def reset_noise(self):
        self.fc_feature_1.reset_noise()
        self.fc_advantage_1.reset_noise()
        self.fc_advantage_2.reset_noise()
        self.fc_value_1.reset_noise()
        self.fc_value_2.reset_noise()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class ConvQNetwork(nn.Module):
    """Convolutional Actor (Policy) Model.
    """

    def __init__(self, num_input_chnl, action_size, seed, num_filters = [16,32], fc_layers=[64,64]):
        """Initialize parameters and build model.

        Args:
            num_input_chnl (int): Number of input channels
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ConvQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(num_input_chnl, num_filters[0], kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv1bnorm = nn.BatchNorm2d(num_filters[0])
        self.conv1relu = nn.ReLU()
        self.conv1maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv2bnorm = nn.BatchNorm2d(num_filters[1])
        self.conv2relu = nn.ReLU()
        self.conv2maxp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = nn.Linear(num_filters[1]*21*21, fc_layers[0])
        self.fc1bnorm = nn.BatchNorm1d(fc_layers[0])
        self.fc1relu = nn.ReLU()

        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc2bnorm = nn.BatchNorm1d(fc_layers[1])
        self.fc2relu = nn.ReLU()

        self.fc3 = nn.Linear(fc_layers[1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
        """

        state = self.conv1(state)
        state = self.conv1bnorm(state)
        state = self.conv1relu(state)
        state = self.conv1maxp(state)

        state = self.conv2(state)
        state = self.conv2bnorm(state)
        state = self.conv2relu(state)
        state = self.conv2maxp(state)

        state = state.reshape((-1,32*21*21))

        state = self.fc1(state)
        state = self.fc1bnorm(state)
        state = self.fc1relu(state)

        state = self.fc2(state)
        state = self.fc2bnorm(state)
        state = self.fc2relu(state)

        state = self.fc3(state)

        return state



