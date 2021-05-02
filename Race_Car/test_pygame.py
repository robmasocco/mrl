import sys
import numpy as np
import pygame

BLACK = (0, 0, 0)
GREY = (50, 50, 50)
WHITE = (200, 200, 200)
WIDTH = 850
HEIGHT = 850
W_CELLS = 30
H_CELLS = 30
MAX_LIM = 6


def generate_map():
    map = np.zeros((W_CELLS, H_CELLS), dtype=np.int32)
    map[W_CELLS // 2:, H_CELLS // 2:] = 1
    lims = np.random.randint(MAX_LIM, size=4)
    for h in range(H_CELLS):
        lims[0] = np.amax(
            [0, np.amin([MAX_LIM, lims[0] + np.random.choice([-1, 0, 1])])])
        map[h, :lims[0]] = 1
        if h > H_CELLS // 2:
            lims[1] = np.amax(
                [0, np.amin([MAX_LIM, lims[1] + np.random.choice([-1, 0, 1])])])
            map[h, H_CELLS // 2 - lims[1]:] = 1
    for w in range(W_CELLS):
        lims[2] = np.amax(
            [0, np.amin([MAX_LIM, lims[2] + np.random.choice([-1, 0, 1])])])
        map[:lims[2], w] = 1
        if w > W_CELLS // 2:
            lims[3] = np.amax(
                [0, np.amin([MAX_LIM, lims[3] + np.random.choice([-1, 0, 1])])])
            map[W_CELLS // 2 - lims[3]:, w] = 1
    return map


def draw_map(map):
    w_size = WIDTH // W_CELLS
    h_size = HEIGHT // H_CELLS
    for h in range(H_CELLS):
        for w in range(W_CELLS):
            rect = pygame.Rect(w * w_size, h * h_size, w_size, h_size)
            if map[h, w] == 0:
                pygame.draw.rect(SCREEN, WHITE, rect, 0)
            else:
                pygame.draw.rect(SCREEN, GREY, rect, 0)
            pygame.draw.rect(SCREEN, BLACK, rect, 1)


if __name__ == '__main__':
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(WHITE)
    map = generate_map()
    while True:
        draw_map(map)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
