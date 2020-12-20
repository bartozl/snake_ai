import sys

import numpy as np
import pygame


class world:
    def __init__(self, height=10, width=10, pixels=400, update_speed=5):
        self.H = height
        self.W = width
        self.clean_grid = np.zeros((self.H, self.W), dtype=np.int)
        self.curr_grid = self.clean_grid.copy()

        pygame.init()
        self.pixels = pixels // self.H
        self.margin = 1
        self.windows_size = np.asarray([self.H, self.W]) * self.pixels + \
                            np.asarray([self.margin * self.H, self.margin * self.W])
        self.screen = pygame.display.set_mode(self.windows_size)
        self.update_speed = update_speed
        self.clock = pygame.time.Clock()
        self.caption = 'snake!'
        pygame.display.set_caption(self.caption)

        self.snake_body = [(0, 0)]
        self.snake_head = self.snake_body[0]
        self.apple = (2, 2)
        self.direction = 'RIGHT'

        self.control = {"UP": 82, "DOWN": 81, "LEFT": 80, "RIGHT": 79}
        self.color = {"WHITE": (255, 255, 255), "GREEN": (35, 250, 44), "RED": (220, 0, 0), "BLACK": (0, 0, 0)}

    def update_grid(self):
        self.curr_grid = self.clean_grid.copy()
        for pos in self.snake_body:
            self.curr_grid[pos] = 1
        self.curr_grid[self.apple] = 2

    def update_screen(self):
        self.screen.fill(self.color['BLACK'])
        self.update_grid()
        self.draw()
        pygame.display.set_caption(self.caption)
        pygame.display.flip()
        self.clock.tick(self.update_speed)

    def draw(self):
        y = self.margin
        for row in self.curr_grid:
            x = self.margin
            for elem in row:
                color = self.color['WHITE']
                if elem == 1:
                    color = self.color['GREEN']
                elif elem == 2:
                    color = self.color['RED']
                pygame.draw.rect(self.screen, color, [x, y, self.pixels, self.pixels])
                x += self.pixels + self.margin
            y += self.pixels + self.margin

    def capture_events(self):
        lost = False
        while not lost:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.scancode == self.control['UP']:
                        # avoid unfeasible events
                        if self.direction not in ['UP', 'DOWN']:
                            self.direction = 'UP'
                    elif event.scancode == self.control['DOWN']:
                        if self.direction not in ['UP', 'DOWN']:
                            self.direction = 'DOWN'
                    elif event.scancode == self.control['LEFT']:
                        if self.direction not in ['LEFT', 'RIGHT']:
                            self.direction = 'LEFT'
                    elif event.scancode == self.control['RIGHT']:
                        if self.direction not in ['LEFT', 'RIGHT']:
                            self.direction = 'RIGHT'
            lost = self.step()
            self.update_screen()

    def step(self):
        if self.direction == 'UP':
            check_head = (self.snake_head[0] - 1, self.snake_head[1])
        elif self.direction == 'DOWN':
            check_head = (self.snake_head[0] + 1, self.snake_head[1])
        elif self.direction == 'LEFT':
            check_head = (self.snake_head[0], self.snake_head[1] - 1)
        elif self.direction == 'RIGHT':
            check_head = (self.snake_head[0], self.snake_head[1] + 1)
        else:
            print('unknown direction')
            sys.exit(1)

        # check if a wall has been touched or if the body has been hit by the head
        if (check_head[0] < 0) or (check_head[0] >= self.H) or \
           (check_head[1] < 0) or (check_head[1] >= self.W) or \
           (check_head in self.snake_body[1:]):
            print(check_head, self.snake_body)
            self.caption = 'OPS! You lose!'
            return True
        else:
            self.snake_head = check_head
            tail = self.snake_body[-1]
            self.snake_body[1:] = self.snake_body[:-1]
            self.snake_body[0] = self.snake_head

            # if the apple has been eaten, make the body longer and change the apple position
            if self.snake_head == self.apple:
                self.snake_body.append(tail)
                self.apple = self.place_apple()

            return False

    def place_apple(self):
        found = False
        while not found:
            x = np.random.randint(0, self.H)
            y = np.random.randint(0, self.W)
            if (x, y) not in self.snake_body:
                found = True
        return (x, y)

    def run(self):
        while True:
            self.__init__()
            self.capture_events()


run_it = world()
run_it.run()
