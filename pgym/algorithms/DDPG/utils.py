from typing import Optional, Union

import gym
import numpy as np
import pgym
import torch
from torch import nn


def combined_shape(length: int, shape: Optional[Union[int, list]] = None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


# Generic replay buffer for standard gym tasks
class StandardBuffer(object):
    def __init__(self, state_dim, action_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.buffer_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done, episode_done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(
            max(0, self.crt_size - self.buffer_size), self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy",
                self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(
            f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(
            f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(
            f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(
            f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")

class RescaleObservation(gym.ObservationWrapper):
    r"""Rescales the continuous observation space of the environment to a range [a,b].

    Example::

        >>> RescaleObservation(env, a, b).observation_space == Box(a,b)
        True

    """
    def __init__(self, env, a, b):
        assert isinstance(env.observation_space, gym.spaces.Box), (
            "expected Box observation space, got {}".format(type(env.observation_space)))
        assert np.less_equal(a, b).all(), (a, b)
        super().__init__(env)
        self.a = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + a
        self.b = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + b
        self.observation_space = gym.spaces.Box(low=a, high=b, shape=env.observation_space.shape, dtype=env.observation_space.dtype)

    def observation(self, observation):
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        observation = self.a + (self.b - self.a)*((observation - low)/(high - low))
        return observation
