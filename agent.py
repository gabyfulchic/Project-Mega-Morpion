#!/usr/bin/env python
import random
import numpy as np

from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark

env = TicTacToeEnv()

try: 
    Q = np.load("table.npy")
except:
    Q = np.zeros((env.observation_space.n, env.action_space.n))

class BaseAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action
        return random.choice(ava_actions)

def learn(state, state2, reward, action, action2):
    old_value = Q[state, action]
    learned_value = reward + gamma * Q[state2, action2]
    Q[state, action] = (1 - lr_rate) * old_value + lr_rate * learned_value

def play(max_episode=1000):
    episode = 0
    start_mark = 'O'
    agents = [BaseAgent('O'),
              BaseAgent('X')]

    while episode < max_episode:
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            env.show_turn(True, mark)
            agent = agent_by_mark(agents, mark)
#           action = np.argmax(Q[state, 1])
            action = env.action_space.sample()
            state2, reward, done, info = env.step(action)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state

        # rotate start
        start_mark = next_mark(start_mark)
        episode += 1


if __name__ == '__main__':
    play()
    np.save("table", Q)






