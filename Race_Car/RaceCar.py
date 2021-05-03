
"""
    RACE CAR
"""

import numpy as np


class RaceCar:
    """
        Fuck autopep8
    """

    def __init__(self, map, init_s, pi, epsilon):
        self.map = map
        self.state = init_s
        self.pi = pi
        self.epsilon = epsilon
        self.states = np.empty(0, dtype=np.int32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.int32)

    def set_state(self, coords):
        self.state = np.ravel_multi_index(coords, self.map.shape)

    def map_state(self):
        return np.asarray(np.unravel_index(self.state, self.map.shape))

    def map_action(self, action):
        return np.asarray(np.unravel_index(action, (11, 6))) - [5, 0]

    def drive(self, coords, velocity):
        new_coords = np.zeros(2, dtype=np.int32)
        new_coords[0] = np.amax([np.amin([coords[0] + velocity[0], self.map.shape[0]]), 0])
        new_coords[1] = np.amax([np.amin([coords[1] + velocity[1], self.map.shape[1]]), 0])
        return new_coords

    def run(self):
        coords = self.map_state()
        while self.state != -1:
            self.states = np.append(self.states, self.state)
            # Choose action.
            if np.random.choice([0, 1], p=[1.0 - self.epsilon, self.epsilon]) == 0:
                action = self.pi[self.state]
            else:
                action = np.random.randint(66)
            self.actions = np.append(self.actions, action)
            velocity = self.map_action(action)
            coords = self.drive(coords, velocity)
            if map(coords[0], coords[1]) == -1:
                self.state = -1
                self.rewards = np.append(self.rewards, -self.map.shape[0] * self.map.shape[1])
            if map(coords[0], coords[1]) == 1:
                self.state = -1
                self.rewards = np.append(self.rewards, 0)
            else:
                self.set_state(coords)
                self.rewards = np.append(self.rewards, -1)
