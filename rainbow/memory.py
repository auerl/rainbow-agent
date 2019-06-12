#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""memory.py: Memory module containing replay memory classes for DRL agents.
"""

from collections import namedtuple, deque
import torch
import numpy as np
import random
import time

class AugmentedPriorityReplayBuffer(object):
    """Fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, batch_size, device, seed, prio_replay=False, from_pixels=False, image_shape=None):
        """Initialize a ReplayBuffer object.

        Args:

            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self._memory_e = deque(maxlen=buffer_size)
        self._batch_size = batch_size
        self._device = device
        self._prio_replay = prio_replay
        self._augment_state = False

        if from_pixels:
            if not image_shape:
                raise ValueError("When learning from pixels image_shape must be given!")
            self._image_shape = image_shape
            self._augment_state = True

        if self._prio_replay:
            self._memory_p = deque(maxlen=buffer_size)

        fields = ["state", "action", "reward", "next_state", "done"]

        self.experience = namedtuple("Experience", field_names=fields)
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, loss=None):
        """Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self._memory_e.append(e)

        if self._prio_replay:
            p = np.max(self._memory_p) if len(self._memory_p) > 0 else 1.0
            self._memory_p.append(p)

    def sample(self, beta=0.4, probs_alpha=0.6):
        """Randomly sample a batch of experiences from memory.
        """
        total_exp = len(self._memory_e)

        if self._prio_replay:

            prios = np.array(self._memory_p)
            probs  = prios ** probs_alpha
            probs /= probs.sum()

            # Sample the experiences with their probability
            indices = np.random.choice(total_exp, self._batch_size, p=probs)
            exp = [self._memory_e[ind] for ind in indices]

            # compute importance sampling weights
            weights  = (total_exp * probs[indices]) ** -beta
            weights /= weights.max()
            weights  = np.array(weights, dtype=np.float32)
            weights = torch.from_numpy(weights).long().to(self._device)

        else:
            # Sample the experiences with their probability
            indices = np.random.choice(total_exp, self._batch_size)
            exp = [self._memory_e[ind] for ind in indices]

        # in case of on pixel training we want to augment states
        if self._augment_state:
            states, next_states = self._augment_states(exp, indices)
        else:
            states = torch.from_numpy(
                np.vstack([e.state for e in exp if e is not None])).float().to(self._device)
            next_states = torch.from_numpy(
                np.vstack([e.next_state for e in exp if e is not None])).float().to(self._device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in exp if e is not None])).long().to(self._device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in exp if e is not None])).float().to(self._device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in exp if e is not None]).astype(np.uint8)).float().to(self._device)

        if self._prio_replay:
            return (states, actions, rewards, next_states, dones, indices, weights)
        else:
            return (states, actions, rewards, next_states, dones, indices)

    def _augment_states(self, exp, indices):
        """
        """
        aug_next_states = []
        aug_states = []

        for i, exp in enumerate(exp):
            ind = indices[i]

            if exp is None or (ind-2) < 0 or (ind+1) >= len(self._memory_e):
                continue

            prev_exp      = self._memory_e[ind-1]
            prev_prev_exp = self._memory_e[ind-2]
            next_exp      = self._memory_e[ind+1]

            #e.state and e.next_state is in Nx3xHxW format (augment state in the C dimension)
            exp_action           = np.ones((1,1,self._image_shape[0],self._image_shape[1])) * exp.action
            prev_exp_action      = np.ones((1,1,self._image_shape[0],self._image_shape[1])) * prev_exp.action
            prev_prev_exp_action = np.ones((1,1,self._image_shape[0],self._image_shape[1])) * prev_prev_exp.action

            aug_state = np.concatenate((prev_prev_exp.state, prev_prev_exp_action, prev_exp.state, prev_exp_action, exp.state), axis=1)
            aug_next_state = np.concatenate((prev_exp.state, prev_exp_action, exp.state, exp_action, next_exp.state), axis=1)

            aug_states.append(aug_state)
            aug_next_states.append(aug_next_state)

        return aug_states, aug_next_states

    def update_priorities(self, indices, prios):
        """Update the priorities using the TD-loss
        """
        for i, prio in zip(indices, prios):
            self._memory_p[i] = prio
        # norm = np.sum(np.array(self._memory_p))
        # for i, _ in enumerate(self._memory_p):
        #     self._memory_p[i] /= norm

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self._memory_e)


