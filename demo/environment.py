import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np


class Environment(object):

    def __init__(self):
        pass


class LoopEnv(Environment):
    """
    The environment defines 'locations' which are states and choose one of them as the next state
    based on a weighted random choice

    """
    def __init__(self, state_dim, n_locations, actions):
        """

        :param state_dim: vector size
        :param n_locations: number of different states
        :param actions: list of possible actions
        """
        super().__init__()

        self.state_dim = state_dim
        self.n_locations = n_locations
        self.actions = actions

        self.locations = np.random.rand(n_locations, state_dim)
        self.locations_weights = np.random.rand(n_locations)
        self.locations_weights /= np.sum(self.locations_weights)
        self.state_m1 = np.random.rand(state_dim)

    def run(self):
        action = np.random.choice(self.actions)

        state_location = np.random.choice(range(self.n_locations), p=self.locations_weights)
        state = self.locations[state_location]

        sequence = (self.state_m1, action, state)
        self.state_m1 = state

        return sequence
