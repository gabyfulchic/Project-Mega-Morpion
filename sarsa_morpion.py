# -*- coding: utf-8 -*-

import gym
import random
import numpy as np
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, after_action_state
import time, pickle, os

env = TicTacToeEnv()
env.reset()
epsilon = 0.9
total_episodes = 10000
max_steps = 100
lr_rate = 0.81
gamma = 0.96

#class BaseAgent():
#    def __ini__(self, mark):
#        self.mark = mark
#
#    def act(self, state, potential_actions):
#        for action in potential_actions:
#            next_state = after_action_state(state, action)
#            actual_status_game = check_game_status(next_state[0])
#            if actual_status_game > 0:
#                 if tomark(actual_status_game) == self.marl:
#                     return action
#        return random.choice(
            

def choose_action(state):
    # action=0
    # if np.random.uniform(0, 1) < epsilon:
    #     action = env.action_space.sample()
    # else:
    action = np.argmax(Q[state, :])
    return action

def learn(state, reward, action):
    old_value = Q[state, action]
    learned_value = reward + gamma * Q[state, action]
    Q[state, action] = (1 - lr_rate) * old_value +  lr_rate * learned_value


Q = np.zeros((env.observation_space.n, env.action_space.n))

# Start
for episode in range(total_episodes):
    start_mark = 'O'
    env.set_start_mark(start_mark)
    state = env.reset()
    _, mark = state
    t = 0
    while t < max_steps:
        env.render()
        potential_actions = env.available_actions()
        action = choose_action(state)
        learn(state, reward, action)
        t += 1
        if done:
            break

print(Q)

np.save("qtable", Q)



Q = np.load("qtable.npy")

"""Evaluate agent's performance after Q-learning"""
def evaluate():
  total_epochs, total_penalties = 0, 0

  for _ in range(total_episodes):
      state = env.reset()
      epochs, penalties, reward = 0, 0, 0
      
      done = False
      
      while not done:
          action = np.argmax(Q[state, :])
          state, reward, done, info = env.step(action)

          if reward < 0:
              penalties += 1

          epochs += 1

      total_penalties += penalties
      total_epochs += epochs

  print(f"Results after {total_episodes} episodes:")
  print(f"Average timesteps per episode: {total_epochs / total_episodes}")
  print(f"Average penalties per episode: {total_penalties / total_episodes}")

evaluate()

