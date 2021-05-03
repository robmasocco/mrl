
"""
    RACE CAR
"""

import numpy as np
import pygame


class RaceCar:
    """
        Fuck autopep8
    """

    def __init__(self, race_map, init_coords, pi, epsilon, width, height):
        self.map = race_map
        self.coords = init_coords
        self.set_state()
        print(self.state)
        self.pi = pi
        self.epsilon = epsilon
        self.states = np.empty(0, dtype=np.int32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.int32)
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((255, 255, 255))
        self.clock = pygame.time.Clock()

    def set_state(self):
        self.state = np.ravel_multi_index(self.coords, self.map.shape)

    def map_state(self):
        return np.asarray(np.unravel_index(self.state, self.map.shape))

    def map_action(self, action):
        return np.asarray(np.unravel_index(action, (11, 6))) - [5, 0]

    def drive(self, velocity):
        self.coords[0] = np.amax(
            [np.amin([self.coords[0] + velocity[0], self.map.shape[0] - 1]), 0])
        self.coords[1] = np.amax(
            [np.amin([self.coords[1] + velocity[1], self.map.shape[1] - 1]), 0])

    def run(self):
        self.draw_map()
        pygame.display.update()
        while self.state != -1:
            self.states = np.append(self.states, self.state)
            # Choose action.
            if np.random.choice([0, 1], p=[1.0 - self.epsilon, self.epsilon]) == 0:
                action = self.pi[self.state]
            else:
                action = np.random.randint(66)
            self.actions = np.append(self.actions, action)
            velocity = self.map_action(action)
            self.drive(velocity)
            if self.map[self.coords[0], self.coords[1]] == -1:
                self.state = -1
                self.rewards = np.append(
                    self.rewards, -self.map.shape[0] * self.map.shape[1])
            elif self.map[self.coords[0], self.coords[1]] == 2:
                self.state = -1
                self.rewards = np.append(self.rewards, 0)
            else:
                self.set_state()
                self.rewards = np.append(self.rewards, -1)
            self.draw_map()
            pygame.display.update()
        pygame.quit()

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_rewards(self):
        return self.rewards

    def draw_map(self):
        current_map = np.copy(self.map)
        current_map[self.coords[0], self.coords[1]] = 3
        width, height = pygame.display.get_surface().get_size()
        w_size = width // self.map.shape[0]
        h_size = height // self.map.shape[1]
        for h in range(self.map.shape[1]):
            for w in range(self.map.shape[0]):
                rect = pygame.Rect(w * w_size, h * h_size, w_size, h_size)
                if current_map[h, w] == -1:
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 0)
                elif current_map[h, w] == 1:
                    pygame.draw.rect(self.screen, (0, 200, 0), rect, 0)
                elif current_map[h, w] == 2:
                    pygame.draw.rect(self.screen, (200, 0, 0), rect, 0)
                elif current_map[h, w] == 3:
                    pygame.draw.circle(
                        self.screen, (0, 0, 255), (w * w_size + w_size/2, h * h_size + h_size/2), w_size/2, 0)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 0)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
