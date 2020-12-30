import sys
import numpy as np
import pygame


class World:
    def __init__(self, screen, height, width, margin, pixels, update_speed=8, reward='gradient', game_num=0):
        self.H = height
        self.W = width
        self.clean_grid = np.zeros((self.H, self.W), dtype=np.int)
        self.pixels = pixels
        self.margin = margin

        self.screen = screen
        self.update_speed = update_speed
        self.clock = pygame.time.Clock()

        self.snake_body = [(self.H // 2, self.W // 2)]
        self.snake_head = self.snake_body[0]
        self.reward_type = reward
        self.apple = self.place_apple()
        self.direction = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.curr_grid = self.clean_grid.copy()
        self.update_world()

        self.actions = {"UP": [1, 0, 0, 0], "DOWN": [0, 1, 0, 0], "LEFT": [0, 0, 1, 0], "RIGHT": [0, 0, 0, 1]}
        self.control = {"UP": 82, "DOWN": 81, "LEFT": 80, "RIGHT": 79}
        self.color = {"WHITE": (255, 255, 255), "GREEN": (35, 250, 44), "RED": (220, 0, 0), "BLACK": (0, 0, 0)}
        self.reward = self.set_reward()
        self.score = 0
        self.steps = 0
        self.game_num = game_num

        self.caption = 'snake! - game {} - score  {} - '.format(game_num, self.score)
        pygame.display.set_caption(self.caption)

    def set_reward(self):
        if self.reward_type == 'discrete':
            reward = {'lost': -10, 'step': np.ones((self.H, self.W)) * -3, 'eat': 10}
        elif self.reward_type == 'gradient':
            grid = None
            if self.apple != -1:  # the game is not lost
                x, y = self.apple
                A_x = list(np.arange(self.H))
                B_x = A_x[:-1][-x:] if x != 0 else []
                C_x = A_x[::-1][:(self.H - x)]

                A_y = list(np.arange(self.W))
                B_y = A_y[:-1][-y:] if y != 0 else []
                C_y = A_y[::-1][:(self.W - y)]

                xx = B_x + C_x
                yy = B_y + C_y

                grid = np.outer(xx, yy) / ((self.H - 1) * (self.W - 1)) - np.ones((self.H, self.W))
            reward = {'lost': -10, 'step': grid, 'eat': 10}
        else:
            raise NotImplementedError
        return reward

    def update_world(self):
        self.curr_grid = self.clean_grid.copy()
        for pos in self.snake_body:
            self.curr_grid[pos] = 1
        self.curr_grid[self.apple] = 2

    def show_world(self):
        pygame.display.set_caption(self.caption)
        pygame.display.flip()
        self.clock.tick(self.update_speed)

    def draw_world(self):
        self.screen.fill(self.color['BLACK'])
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

    def change_direction(self, action_code):
        # action_code is in the form [1, 0, 0, 0]
        action_name = list(self.actions.keys())[list(action_code).index(1)]

        if action_name == 'UP':
            if self.direction not in ['UP', 'DOWN']:
                self.direction = 'UP'

        elif action_name == 'DOWN':
            if self.direction not in ['UP', 'DOWN']:
                self.direction = 'DOWN'

        elif action_name == 'LEFT':
            if self.direction not in ['LEFT', 'RIGHT']:
                self.direction = 'LEFT'

        elif action_name == 'RIGHT':
            if self.direction not in ['LEFT', 'RIGHT']:
                self.direction = 'RIGHT'

    def step(self):
        self.steps += 1
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
            self.caption = 'OPS! You lose!'
            return self.reward['lost']
        else:
            self.snake_head = check_head
            tail = self.snake_body[-1]
            self.snake_body[1:] = self.snake_body[:-1]
            self.snake_body[0] = self.snake_head
            reward = self.reward['step'][self.snake_head]

            # if the apple has been eaten, make the body longer and change the apple position
            if self.snake_head == self.apple:
                self.score += 1
                self.caption = 'snake! - game {} - score  {} - '.format(self.game_num, self.score)
                self.snake_body.append(tail)
                self.apple = self.place_apple()
                self.reward = self.set_reward()
                if self.apple == -1:
                    # the snake has reached the maximum length, the game is over.
                    reward = self.reward['lost']
                else:
                    reward = self.reward['eat']
            return reward

    def place_apple(self):
        candidates = [(x, y) for x in range(self.H) for y in range(self.W) if (x, y) not in self.snake_body]
        if len(candidates) == 0:
            return -1
        else:
            return candidates[np.random.randint(len(candidates))]

    """
    def capture_events(self):
        reward = 0
        while reward != self.reward['lost']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    for action_name in self.control:  # action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                        if event.scancode == self.control[action_name]:
                            action_code = self.actions[action_name]
                            self.change_direction(action_code)
            reward = self.step()
            self.update_world()
            if self.screen is not None:
                self.draw_world()
                self.show_world()

    def play(self):
        self.capture_events()
    """
