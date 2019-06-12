#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""agents.py: Agents module, containing different DRL agents."""

import numpy as np
import random
from collections import namedtuple, deque

# DRL related
from .models import QNetwork, DuelingQNetwork
from .models import NoisyQNetwork, NoisyDuelingQNetwork
from .memory import AugmentedPriorityReplayBuffer

# For Pixel training
from .models import ConvQNetwork

# PyTorch related
import torch
import torch.nn.functional as F
import torch.optim as optim


import time

class RainbowAgent(object):
    """Implementation of the Rainbow Algorithm as of:

    ``Rainbow: Combining Improvements in Deep Reinforcement Learning``,
    M. Hessel, J. Modayil, H. v. Hasselt, T. Schaul, G. Ostrovski,
    W. Dabney, D. Horgan, B. Piot, M. Azar, D. Silver (2017).

    This agent is based on PyTorch and several reference Neural Network
    architectures. The learning module combines the following approaches
    to the original DQN algorithm:

    1. Double Q-learning
    2. Prioritized Experience Replay
    3. Dueling Networks
    4. Noisy Networks
    5. Multi-step learning (TODO)
    6. Distributional RL (TODO)

    Interacts with and learns from a provided environment. The environment
    can either be a OpenAI Gym environment or a UnityEnvironment object.
    Can learn from pixels, using a ConvNet or from state space described
    other representations.
    """

    DEFAULT_PARAMS = {
        # learning
        "gamma":              0.99, # discount factor
        "tau":                1e-3, # soft update factor
        "lr":                 5e-4, # learning rate
        "update_every":          4, # how often to update the network
        # from pixel learning
        "from_pixels":       False, # learn from pixels using ConvNet
        "image_shape":     (84,84), # shape of the image to learn from
        "input_channels":       11, # number of input channels of augmented state
        # memory
        "buffer_size":    int(1e5), # replay memory buffer size
        "batch_size":           64, # replay memory batch size
        "priority_replay":    True, # use prioritized experience replay
        "augment_state":      True,
        # network
        "double_dqn":         True, # use Double DQN
        "dueling_nets":       True, # use Dueling Network
        "noisy_nets":        False, # use Noisy Network
    }

    DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self, state_size, action_size, seed, params=None, device=None):
        """Initialize an Agent object.

        Args:
            state_size (int): Dimension of the state state.
            action_size (int): Dimension of the action space.
            seed (int): Random number seed for replay memory and neural networks.
            params (Dict, optional): Param dictionary to be used to initialize agent.
            device (torch.device, optional): The device to by used by torch.

        Attributes:

            .. Rainbow components:
            _prio_replay (bool): Turn on prioritized experience replay and importance sampling.
            _double_dqn (bool): Turn on the double DQN algorithm.
            _dueling_nets (bool): Turn on dueling network architecture.
            _noisy_nets (bool): Turn on noisy neural networks.

            .. Algorithm parameters:
            _gamma (float): The discount factor
            _tau (float): The soft-update factor.
            _lr (float): The learning rate.
            _update_every (int): Update local network every _update_every timestep.

            .. Replay memory parameters:
            _buffer_size (int): Replay memory buffer size.
            _batch_size (int): Replay memory batch size.

            .. Visual Learning parameters:
            _from_pixels (bool): Turns on using network architectures with convolutional layers.
            _image_shape (tuple): Tuple defining the image dimension
            _input_channels (int): Number of input channels in augmented state

            .. Internal variables
            _t_step (int): The current timestep.
            _seed (int): Random number generator seed.

            .. Public variables:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.

        """
        if not device:
            self._device = self.DEFAULT_DEVICE
        if not params:
            self._params = self.DEFAULT_PARAMS
        else:
            self._params = params

        # Public variables
        self.state_size = state_size
        self.action_size = action_size

        self._seed = seed

        # Learning config
        self._tau           = self._params['tau']
        self._gamma         = self._params['gamma']
        self._update_every  = self._params['update_every']
        self._lr            = self._params['lr']

        # Network config
        self._noisy_nets    = self._params['noisy_nets']
        self._dueling_nets  = self._params['dueling_nets']
        self._double_dqn    = self._params['double_dqn']

        # Replay memory config
        self._buffer_size   = self._params['buffer_size']
        self._batch_size    = self._params['batch_size']
        self._prio_replay   = self._params['priority_replay']

        # Learning from pixels
        self._from_pixels   = self._params['from_pixels']
        if self._from_pixels:
            if self.state_size:
                raise ValueError("State size parameter must not be set when learning from pixels")
            self._image_shape = self._params['image_shape']
            self._input_channels = self._params['input_channels']
        else:
            self._image_shape = None
            self._input_channels = None

        # Based on the network config, set up PyTorch neural nets
        if self._from_pixels:
            self._setup_visual_networks()
        else:
            self._setup_networks()

        # Setup optimizer
        self._optimizer = optim.Adam(self._qnetwork_local.parameters(), lr=self._lr)

        # Replay memory
        self._memory = AugmentedPriorityReplayBuffer(
            self._buffer_size,
            self._batch_size,
            self._device,
            self._seed,
            self._prio_replay, # turns on priority replay if requested
            self._from_pixels,
            self._image_shape
        )

        # Initialize time step (for updating every `_update_every` steps)
        self._t_step = 0

    def load_state_dict(self, state_dict):
        """Populates the network parameters from a checkpoint state dict.

        Args:
            state_dict (Dict): A dictionary containing state of
                toch.nn.module.
        """
        self._qnetwork_local.load_state_dict(state_dict)

    def get_state_dict(self):
        """Returns the current state_dict.

        Returns:
            state_dict (Dict): A dictionary containing state of
                toch.nn.module.
        """
        return self._qnetwork_local.state_dict()

    def augment_state(self, state):
         """Augment the state to include previous observations and actions.
         Used in case we learn from pixels.
         """

         shp = self._image_shape

         if len(self._memory) >= 2:
             prev_idx = len(self._memory)-1
             prev_prev_idx = prev_idx-1
             prev_e = self._memory._memory_e[prev_idx]
             prev_prev_e = self._memory._memory_e[prev_prev_idx]

             #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
             prev_e_a = np.ones((1,1,shp[0],shp[1]))*prev_e.action
             prev_prev_e_a = np.ones((1,1,shp[0],shp[1]))*prev_prev_e.action
             aug_state = np.concatenate((prev_prev_e.state, prev_prev_e_a, prev_e.state, prev_e_a, state), axis=1)
         else:
             #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
             initial_action = 0
             prev_e_a = np.ones((1,1,shp[0],shp[1]))*initial_action
             prev_prev_e_a = np.ones((1,1,shp[0],shp[1]))*initial_action
             aug_state = np.concatenate((state, prev_prev_e_a, state, prev_e_a, state), axis=1)


         return aug_state

    def step(self, state, action, reward, next_state, done, beta=0.4):
        """Method adding state, action, reward tuples to the experience buffer
        and, every UPDATE_EVERY, runs the learn function which will perform
        gradient descent on the target network and subsequently update the
        local network.

        Args:
            state (numpy.ndarray): Array giving the state at the current timestep.
            action (numpy.ndarray): The action perform at the current state.
            reward (numpy.ndarray): The reward earned between the current and the
                next timesteps.
            next_state (numpy.ndarray): Array giving state at the next timestep.
            done (bool): Boolean saying whether the episode is done.
        """
        # Save frame in replay memory
        self._memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self._t_step = (self._t_step + 1) % self._update_every

        if self._t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self._memory) > self._batch_size:
                experiences = self._memory.sample(beta)
                self.learn(experiences, self._gamma)

    def act(self, state, eps=0.):
        """Returns actions for provided state as per current policies given by the
        local Q-Network.

        With a probability of 1-eps the greedy action will be chosen, with a
        probability of eps, a random action will be used.
        
        Args:
            state (array_like): Current state.
            eps (float, optional): Epsilon for epsilon-greedy action selection.
                Needs to be between 0 and 1.

        Raises:
            ValueError: Raised when eps is not between 0 and 1.

        Returns:
            numpy.ndarray: Numpy array containing selected actions.
        """
        if eps < 0. or eps > 1.:
            raise ValueError('Epsilon not between 0 and 1!')

        if self._from_pixels:
            state = torch.from_numpy(state).float().to(self._device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)

        # Sets the module in evaluation mode
        self._qnetwork_local.eval()

        # Context-manager that disables gradient calculation (this is possible
        # when no tensor.backward() is called and reduces memory consumption).
        with torch.no_grad():

            # Use the local network to get action values
            action_values = self._qnetwork_local(state)

        # Set the module to training mode
        self._qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self._prio_replay:
            states, actions, rewards, next_states, dones, indices, weights  = experiences
        else:
            states, actions, rewards, next_states, dones, indices = experiences

        # Get max predicted Q values (for next states) from target model
        if self._double_dqn:
            q_network = self._qnetwork_target
        else:
            q_network = self._qnetwork_local

        q_targets_next = q_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self._qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # In case importance sampling is turned on
        if self._prio_replay:
            prios  = (q_targets - q_expected.detach()).pow(2) * torch.tensor(
                weights, requires_grad=True).float().to(self._device) + 1e-5
            self._memory.update_priorities(indices, prios.data.cpu().numpy()[0])

        # Minimize loss
        self._minimize_loss(loss)

        # Reset noise, if noisy nets are turned on
        if self._noisy_nets:
            self._reset_noise()

        # Update target network: After each training step we copy over the parameters
        # from the target to the local network.
        if self._double_dqn:
            self._soft_update(
                self._qnetwork_local,
                self._qnetwork_target,
                self._tau
            )

    def _minimize_loss(self, loss):
        """Minimizes the loss.

        Args:
            loss (torch.Tensor): Element-wise mean squared error.
        """
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _reset_noise(self):
        """Resets the noise, when a noisy network is used.
        """
        self._qnetwork_local.reset_noise()
        if self._double_dqn:
            self._qnetwork_target.reset_noise()

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (torch.nn.Module): PyTorch Model from which weights copied.
            target_model (torch.nn.Module): PyTorch Model to which Weights will be copied.
            tau (float): Interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _setup_visual_networks(self):
        """Setup the conv network configuration, depending on
        the chosen configuration. Note that Dueling and Noisy
        networks are not yet implemented for visual learning.
        """
        seed = self._seed
        ns = self._input_channels
        na = self.action_size

        self._qnetwork_local = ConvQNetwork(
            ns, na, seed).to(self._device)
        self._qnetwork_target = ConvQNetwork(
            ns, na, seed).to(self._device)


    def _setup_networks(self):
        """Setup the network configuration, depending on
        the chosen configuration.
        """
        seed = self._seed
        ns = self.state_size
        na = self.action_size

        # Setup Q-Networks
        if self._double_dqn:
            if self._dueling_nets:
                if self._noisy_nets:
                    self._qnetwork_local = NoisyDuelingQNetwork(
                        ns, na, seed).to(self._device)
                    self._qnetwork_target = NoisyDuelingQNetwork(
                        ns, na, seed).to(self._device)
                else:
                    self._qnetwork_local = DuelingQNetwork(
                        ns, na, seed).to(self._device)
                    self._qnetwork_target = DuelingQNetwork(
                        ns, na, seed).to(self._device)
            else:
                if self._noisy_nets:
                    self._qnetwork_local = NoisyQNetwork(
                        ns, na, seed).to(self._device)
                    self._qnetwork_target = NoisyQNetwork(
                        ns, na, seed).to(self._device)
                else:
                    self._qnetwork_local = QNetwork(
                        ns, na, seed).to(self._device)
                    self._qnetwork_target = QNetwork(
                        ns, na, seed).to(self._device)

        # In case this is not a double dqn
        else:
            if self._dueling_nets:
                if self._noisy_nets:
                    self._qnetwork_local = NoisyDuelingQNetwork(
                        ns, na, seed).to(self._device)
                else:
                    self._qnetwork_local = DuelingQNetwork(
                        ns, na, seed).to(self._device)
            else:
                if self._noisy_nets:
                    self._qnetwork_local = NoisyQNetwork(
                        ns, na, seed).to(self._device)
                else:
                    self._qnetwork_local = QNetwork(
                        ns, na, seed).to(self._device)
