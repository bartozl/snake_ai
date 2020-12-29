import argparse
import world
from q_learn_utils import create_input, choose_action, execute_action, compute_Q_target, compute_epsilon, save_infos, \
    load_infos
from network import cnn
import pygame
import numpy as np
from tensorflow import keras
from collections import defaultdict
from pathlib import Path
from time import time
from datetime import timedelta
import json

# TODO create a config file instead of parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-P', type=int, default=400)
parser.add_argument('-H', type=int, default=4)
parser.add_argument('-W', type=int, default=4)
parser.add_argument('-M', type=int, default=1)
parser.add_argument('-R', type=str, default='gradient')

parser.add_argument('-epsilon', type=str, default='sinusoidal')
parser.add_argument('-gamma', type=float, default=0.95)
parser.add_argument('-n_games', type=int, default=20000)
parser.add_argument('-observe', type=float, default=0.25)
parser.add_argument('-bs', type=int, default=256)
parser.add_argument('-channels', type=int, default=4)
parser.add_argument('-len_mem', type=int, default=1000)  # steps
parser.add_argument('-augment', type=int, default=5)

parser.add_argument('-save_every', type=int, default=0.1)
parser.add_argument('-train', action='store_true', default=False)
parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-show', action='store_true', default=False)
parser.add_argument('-run', type=int, default=0)

args = parser.parse_args()

run_path = f'./run_{args.run}'
Path(f'{run_path}/memory/').mkdir(parents=True, exist_ok=True)
with open(f'{run_path}/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

height, width, margin = args.H, args.W, args.M
pixels = args.P // height
assert 0 <= args.observe <= 1, "The observe argument is the percentage of completely random moves. Must be in [0, 1]"

if args.show or args.test:
    screen_h = height * pixels + margin * height
    screen_w = width * pixels + margin * height
    screen = pygame.display.set_mode((screen_h, screen_w))
else:
    screen = None

model = cnn(input_shape=(args.channels, height * args.augment, width * args.augment), num_actions=4)
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam())

checkpoint, game, memory = load_infos(run_path)
if checkpoint is not None:
    model.load_weights(checkpoint)

state_curr = None
snake_world = world.World(screen, height, width, margin, pixels, reward=args.R, game_num=game)

if args.train:

    n_games = args.n_games
    data = defaultdict(list)
    loss_value = 'not computed yet'
    interval_steps = 1000  # compute the time per step over a mean of 1000 steps
    interval_game = 100  # compute steps per game over a mean of 100 games
    time_per_step = [0] * interval_steps
    step_per_game = [0] * interval_game
    epsilon = compute_epsilon(n_games, args.observe, mode=args.epsilon)

    while game < n_games:

        t_start = time()

        state_curr = create_input(snake_world, state_curr, args.augment)
        action = choose_action(model, state_curr, epsilon=epsilon[game])
        reward = execute_action(snake_world, action)
        state_next = (create_input(snake_world, state_curr, args.augment))

        memory.append((state_curr, action, reward, state_next))

        # train the network with a batch number of samples
        if len(memory) == args.len_mem:
            idx = np.random.choice(len(memory), args.bs)
            mini_batch = np.asarray(memory)[idx]  # 64, 4
            state_curr_batch, Q_target_batch = compute_Q_target(mini_batch, model, args.gamma, snake_world.reward['lost'])
            history = model.fit(state_curr_batch, Q_target_batch, batch_size=args.bs, verbose=0)
            loss_value = history.history['loss'][0]
            memory.pop(0)

        time2end = (np.mean(time_per_step)) * np.mean(step_per_game) * (n_games - game)
        log = 'game_num: {:<7} steps: {:<5} score {:<4} epsilon {:<20} loss {:<22} time2end {}'. \
            format(str(game+1), str(snake_world.steps), str(snake_world.score), str(epsilon[game]), str(loss_value),
                   timedelta(seconds=time2end))
        print(log)

        if reward == snake_world.reward['lost']:
            data['steps'].append(snake_world.steps)
            data['score'].append(snake_world.score)
            data['epsilon'].append(epsilon[game])
            data['loss'].append(loss_value if type(loss_value) != str else -1)
            step_per_game.append(snake_world.steps)
            if game != 0 and (game + 1) % int(np.rint(n_games * args.save_every)) == 0:
                model.save_weights(f'{run_path}/checkpoints/weights_{game+1}')
                save_infos(run_path, memory, game + 1, data)
                data = defaultdict(list)
            game += 1
            snake_world = world.World(screen, height, width, margin, pixels, reward=args.R, game_num=game)
            state_curr = None

        if args.show:
            snake_world.draw_world()
            snake_world.show_world()

        time_per_step.append((time() - t_start))
        if len(time_per_step) > interval_steps:
            time_per_step.pop(0)
        if len(step_per_game) > interval_game or len(step_per_game) > (n_games - game):
            step_per_game.pop(0)

if args.test:
    while True:
        state_curr = create_input(snake_world, state_curr, args.augment)
        action = choose_action(model, state_curr, epsilon=0)
        reward = execute_action(snake_world, action)
        if reward == snake_world.reward['lost']:
            snake_world = world.World(screen, height, width, margin, pixels, reward=args.R, game_num=game)
            state_curr = None
        # state_next = (create_input(snake_world, state_curr, args.augment))
        snake_world.draw_world()
        snake_world.show_world()
