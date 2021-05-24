#!/usr/bin/env python3
"""
    File: RaceCar.py
    Brief: Race car simulation class and run methods.
    Authors: Alessandro Tenaglia, Roberto Masocco, Giacomo Solfizi
    Date: May 24, 2021
"""

import numpy as np

class RaceCar:
    """
        Models the Race Car and its runs.
    """

    def __init__(self, race_map, init_coords, pi, epsilon, width, height, actions_x, actions_y):
        """Initialize the Race Car with initial state, dimensions and other metadata."""
        self.map = race_map
        self.coords = init_coords  # These are indexes inside the race map matrix.
        self.actions_x = actions_x
        self.actions_y = actions_y
        self.set_state()  # Call the state setter that uses the local coordinates to set the initial state.
        self.pi = pi
        self.epsilon = epsilon
        self.states = np.empty(0, dtype=np.int32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.int32)

    def set_state(self):
        """Sets the state from the race map matrix indexes."""
        self.state = np.ravel_multi_index(self.coords, self.map.shape)

    def map_state(self):
        """Returns the indexes inside the race map matrix for a given state."""
        return np.asarray(np.unravel_index(self.state, self.map.shape))

    def map_action(self, action):
        """Returns the XY displacements for a given action."""
        return np.asarray(np.unravel_index(action, (self.actions_x, self.actions_y)))

    def drive(self, velocity):
        """
        Drives the car.
        We decided to perform the horizontal movement first and then the vertical one,
        checking the validity of the cells along the path.
        NOTE: Since, in the race map, horizontal movements span columns and vertical ones
        span rows, many indexes might look reversed.
        """
        # Loop over horizontal coordinates updates.
        temp_x = 0
        for x in range(velocity[0] + 1):
            # Update X staying inside the map.
            temp_x = np.amax(
                [np.amin([self.coords[1] + x, self.map.shape[1] - 1]), 0])
            if self.map[self.coords[0], temp_x] == -1:
                # Hit obstacle.
                self.coords[1] = temp_x
                return
        # Hit no obstacle: update coordinates.
        self.coords[1] = temp_x
        # Loop over vertical coordinates updates.
        temp_y = 0
        for y in range(velocity[1] + 1):
            # Update Y staying inside the map. Note that by the matrix's point of view
            # we're climbing i.e. decrementing the coordinate.
            temp_y = np.amax(
                [np.amin([self.coords[0] - y, self.map.shape[0] - 1]), 0])
            if self.map[temp_y, self.coords[1]] == -1:
                # Hit obstacle.
                self.coords[0] = temp_y
                return
        # Hit no obstacle: update coordinates.
        self.coords[0] = temp_y

    def run(self, disp):
        """Runs a race until a wall is hit or the finish line is reached."""
        # Loop until a terminal state is reached.
        while self.state != -1:
            self.states = np.append(self.states, self.state)
            # Choose action according to an eps-greedy policy.
            if np.random.choice([0, 1], p=[1.0 - self.epsilon, self.epsilon]) == 0:
                action = self.pi[self.state]
            else:
                action = np.random.randint(self.actions_x * self.actions_y)
            self.actions = np.append(self.actions, action)
            # Use the action to move the car.
            velocity = self.map_action(action)
            self.drive(velocity)
            # Update the state accordingly.
            if self.map[self.coords[0], self.coords[1]] == -1:
                # Hit obstacle: very low reward.
                self.state = -1
                self.rewards = np.append(self.rewards, -self.map.shape[0] * self.map.shape[1])
            elif self.map[self.coords[0], self.coords[1]] == 2:
                # Reached finish line.
                self.state = -1
                self.rewards = np.append(self.rewards, 0)
            else:
                # Not done yet: reward is -1 since we weigh time.
                self.set_state()
                self.rewards = np.append(self.rewards, -1)

    def get_states(self):
        """Returns visited states."""
        return self.states

    def get_actions(self):
        """Returns performed actions."""
        return self.actions

    def get_rewards(self):
        """Returns earned rewards for each time step."""
        return self.rewards
