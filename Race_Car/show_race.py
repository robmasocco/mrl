#!/usr/bin/env python3
"""
    File: show_race.py
    Brief: Methods to show a race car simulation with saved data.
    Authors: Alessandro Tenaglia, Roberto Masocco, Giacomo Solfizi
    Date: May 24, 2021
"""

import numpy as np
import pygame
from RaceCar import *

def draw_map(coords, width, height, map, screen):
    """Draws the current race map."""
    current_map = np.copy(map)
    current_map[coords[0], coords[1]] = 3
    w_size = width // map.shape[0]
    h_size = height // map.shape[1]
    for h in range(map.shape[1]):
        for w in range(map.shape[0]):
            rect = pygame.Rect(w * w_size, h * h_size, w_size, h_size)
            if current_map[h, w] == -1:
                pygame.draw.rect(screen, (50, 50, 50), rect, 0)
            elif current_map[h, w] == 1:
                pygame.draw.rect(screen, (0, 200, 0), rect, 0)
            elif current_map[h, w] == 2:
                pygame.draw.rect(screen, (200, 0, 0), rect, 0)
            elif current_map[h, w] == 3:
                pygame.draw.circle(screen, (0, 0, 255), (w * w_size + w_size/2, h * h_size + h_size/2), w_size/2, 0)
            else:
                pygame.draw.rect(screen, (200, 200, 200), rect, 0)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

def main():
    """Runs an episode."""
    # Load necessary data.
    pi = np.load("pi.dat", allow_pickle=True)
    race_map = np.load("RaceMap.dat", allow_pickle=True)

    # Initialize simulation data.
    #! These come from the notebook!
    width = 850
    height = 850

    S = race_map.shape[0] * race_map.shape[1]

    actions_x = 6
    actions_y = 6
    A = actions_x * actions_y

    init_states = np.argwhere(race_map == 1)
    init_coords = np.copy(init_states[np.random.randint(init_states.shape[0])])

    # Initialize PyGame.
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    screen.fill((255, 255, 255))
    clock = pygame.time.Clock()

    # Run the episode.
    car = RaceCar(race_map, init_coords, pi, 0.0, actions_x, actions_y)
    car.run()
    states = car.get_states()

    # Show all states.
    for s in states:
        coords = np.asarray(np.unravel_index(s, race_map.shape))
        draw_map(coords, width, height, race_map, screen)
        pygame.display.update()
        clock.tick(1)

    # Done!
    pygame.quit()

if __name__ == "__main__":
    main()
