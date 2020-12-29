import json
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_input(snake_world, state_prev, augment):
    state_curr = snake_world.curr_grid.astype(np.float)  # e.g. shape = [10, 10]
    state_curr *= (1 / np.max(state_curr))  # normalize input
    state_curr = np.kron(state_curr, np.ones((augment, augment)))  # shape = [10 * augment, 10 * augment]
    state_curr = state_curr[np.newaxis, ...]  # shape = [1, 50, 50] if augment = 5

    if state_prev is None:
        state_curr = np.repeat(state_curr, repeats=4, axis=0)  # input.shape = [4, 50, 50]
    else:
        # state_prev.shape = [1, 4, 50, 50]
        state_prev = keras.backend.squeeze(state_prev, axis=0)  # [4, 50, 50]
        # remove the oldest channel and add the current world state in the batch
        state_curr = np.concatenate((state_prev[1:, ...], state_curr), axis=0)  # [4, 50, 50]

    state_curr = state_curr[np.newaxis, ...]  # [1, 4, 50, 50] correct shape for the network
    return keras.backend.constant(state_curr)


def compute_epsilon(n_games, observe=0.2, mode='linear'):
    if mode == 'linear':
        # actions are completely for n_observe steps and then randomness linearly decreases
        epsilon = np.ones(n_games)
        n_observe = int(np.rint(n_games * observe))  # e.g. int(100 * 0.2)
        n_decrement = int(np.rint(n_games * (1 - observe)))
        mask = np.arange(n_observe, n_games)
        epsilon[mask] = np.linspace(1, 0, n_decrement)  # [1, ..., 1, 0.99, 0.98, 0.97, ..., 0]

    if mode == 'sinusoidal':
        assert n_games == 20000, print('The sinusoidal value have been designed for n_games = 25.000')
        X = n_games
        e_d = 0.9995
        x = np.arange(0, X)
        n = X // 800
        epsilon = 0.5 * np.power(e_d, x) * (1 + np.cos(2 * np.pi * n * x / X))

    if mode == 'exp':
        raise NotImplementedError

    return epsilon


def choose_action(model, state, num_actions=4, epsilon=0.5):
    # choose random or reasoned action with a probability that depends on epsilon
    action = np.zeros(num_actions)
    if np.random.rand() < epsilon:
        action[np.random.randint(0, num_actions)] = 1
    else:
        action[np.argmax(model(state))] = 1
    return action


def execute_action(snake_world, action):
    # execute the action and collect the reward
    snake_world.change_direction(action)
    reward = snake_world.step()
    snake_world.update_world()
    return reward


def compute_Q_target(mini_batch, model, gamma, lost_reward):
    # vectorized implementation
    state_curr = np.concatenate(mini_batch[:, 0], axis=0)
    action = np.stack(mini_batch[:, 1], axis=0)  # shape [256, 4]
    reward = mini_batch[:, 2]
    state_next = np.concatenate(mini_batch[:, 3], axis=0)
    Q_state_next = model(state_next)
    Q_target = np.zeros(mini_batch.shape)  # shape [256, 4]
    lost_idx = reward == lost_reward
    Q_target[lost_idx] = action[lost_idx] * lost_reward  # shape [256, 4] in the form [0, -10, 0, 0]
    Q_target[~lost_idx] = action[~lost_idx] * (reward[~lost_idx] + gamma * np.max(Q_state_next[~lost_idx], axis=1))[
        ..., np.newaxis]
    return state_curr, Q_target


def save_infos(run_path, memory, game, data_new, verbose=True):
    mem_path = f'{run_path}/memory/{game}'
    data_path = f'{run_path}/data.json'
    np.save(mem_path, np.asarray(memory))
    print(f'memory saved in {mem_path}')
    try:
        with open(data_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError as e:
        # the file has not been created yet, we create a placeholder to make the method consistent
        data = defaultdict(list)
    for key in data_new:
        data[key] += data_new[key]
    with open(data_path, 'w') as file:
        json.dump(data, file)
    print(f'data saved in {data_path}')


def load_infos(run_path):
    try:
        latest = tf.train.latest_checkpoint(f'{run_path}/checkpoints/')
        game = int(''.join(filter(str.isdigit, latest)))
        memory = list(np.load(f'{run_path}/memory/{game}.npy', allow_pickle=True))
    except (AttributeError, TypeError):
        latest = None
        game = 0
        memory = []
        print('No weights to load')
    return latest, game, memory
