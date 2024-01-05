# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import QLearningAgent
##### START CODING HERE #####
# from visualize import display_frames_as_gif
from gym import wrappers
# from time import time
# This code block is optional. You can import other libraries or define your utility functions if necessary.
##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####
env = wrappers.RecordVideo(env, './videos/')
# construct the intelligent agent.
agent = QLearningAgent(all_actions)

# load the q_table from a file, set epsilon to 0
# agent.load_qtable()
# agent.epsilon = 0

episode_reward_lst = dict()
# initialize the episode reward list
for episode in range(1000):
    episode_reward_lst[episode] = []
episode_state_map = [[0] * 48 for k in range(1000)]
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    episode_reward_lst[episode].append(episode_reward)
    # reset env
    s = env.reset()
    # frames.append(s)
    episode_state_map[episode][s] += 1
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        # env.render()
        # update the episode reward
        episode_reward += r
        episode_reward_lst[episode].append(episode_reward)
        print(f"s = {s}, action = {a}, next state = {s_}, reward = {r}, isdone = {isdone}")
        # agent learns from experience
        agent.learn(state1=s, action1=a, reward=r, state2=s_)
        s = s_
        # frames.append(s)
        episode_state_map[episode][s] += 1
        if isdone:
            time.sleep(0.1)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
print('\ntraining over\n')   
agent.save_qtable()
agent.save_epsilon()
np.save("qlearning/ql_episode_reward_lst.npy", episode_reward_lst)
np.save("qlearning/ql_episode_state_map.npy", episode_state_map)
# close the render window after training.
env.close()

##### END CODING HERE #####


