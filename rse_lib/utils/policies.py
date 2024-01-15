from abc import ABC, abstractmethod

import numpy as np


class AbstractPolicy(ABC):

    @abstractmethod
    def get_action(self, state):
        pass


class FixedPolicy(AbstractPolicy):

    def __init__(self, action):
        self.action = action

    def get_action(self, state):
        return np.ones(1) * self.action


class NeuralPolicy(AbstractPolicy):

    def __init__(self, network):
        self.neural_regressor = network

    def get_action(self, state):
        return self.neural_regressor(state)


class SocPolicy(AbstractPolicy):
    SOC_INDEX = 0

    def __init__(self, min_soc, max_soc, soc_idx, c_rate_idx):
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.soc_index = soc_idx
        self.c_rate_idx = c_rate_idx

    def get_action(self, state):
        soc = state[self.soc_index]
        # positive discharging, negative charging
        c_rate = state[self.c_rate_idx]
        if c_rate < 0 and soc >= self.max_soc:  # charging outside the permitted range
            return np.zeros(1)
        elif c_rate > 0 and soc <= self.min_soc:  # discharging outside the permitted range
            return np.zeros(1)
        else:
            return np.ones(1)
