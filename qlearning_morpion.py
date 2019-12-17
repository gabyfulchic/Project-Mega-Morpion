# -*- coding: utf-8 -*-

import gym
import random
import numpy as np
import gym_tictactoe
import time, pickle, os

#env = gym.make('FrozenLake8x8-v0')
env = gym.make('tictactoe-v0')
env.reset()
epsilon = 0.9
total_episodes = 10000
max_steps = 100
lr_rate = 0.81
gamma = 0.96

def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    old_value = Q[state, action]
    learned_value = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = (1 - lr_rate) * old_value +  lr_rate * learned_value

Q = np.zeros((env.observation_space.n, env.action_space.n))

# Start
for episode in range(total_episodes):
    state = env.reset()
    t = 0
    while t < max_steps:
        #env.render()
        action = choose_action(state)  
        state2, reward, done, info = env.step(action)  
        learn(state, state2, reward, action)
        state = state2
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

