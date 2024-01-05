# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym


##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 0.1  # epsilon-greedy algorithm
        self.epsilon_lst = [self.epsilon]  # for plot
        self.lr = 0.5
        # self.alpha = 0.9  # learning rate
        self.gamma = 0.9  # reward decay
        self.e_decay = 0.99995  # epsilon-decay schema
        self.q_table = np.zeros((48, len(self.all_actions)))  # Q-table
        # self.load_qtable()

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm.
        epsilon probs, choose the a that max q(s,a)
        1 - epsilon probs, choose a random action from self.all_actions
        """
        if np.random.uniform() > self.epsilon:  # epsilon probs, np.random.uniform() return a random float in the interval [0.0, 1.0)
            # choose the action with max q value
            action = np.argmax(self.q_table[observation, :])
        else:
            # choose a random action
            action = np.random.choice(self.all_actions)
        self.epsilon = self.epsilon * self.e_decay  # epsilon decay
        self.epsilon_lst.append(self.epsilon)
        return action

    def learn(self, state1, action1, reward, state2, action2):
        """learn from experience"""
        print("I should learn! (ﾉ｀⊿´)ﾉ, epsilon: ", self.epsilon, "e_decay: ", self.e_decay, "lr: ", self.lr, "gamma: ", self.gamma)
        self.q_table[state1][action1] = self.q_table[state1][action1] + self.lr * (
                    reward + self.gamma * self.q_table[state2][action2] - self.q_table[state1][action1])
        return False

    def save_qtable(self):
        """You can add other functions as you wish."""
        # save the q_table to a file
        np.savetxt("sarsa/sarsa_q_table.txt", self.q_table)

        return None

    def save_epsilon(self):
        """You can add other functions as you wish."""
        # save the epsilon to a file
        np.savetxt("sarsa/sarsa_epsilon.txt", self.epsilon_lst)

        return None

    def load_qtable(self):
        """You can add other functions as you wish."""
        # load the q_table from a file
        self.q_table = np.loadtxt("sarsa/sarsa_q_table.txt")
        return None
    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 0.8  # epsilon-greedy algorithm
        self.epsilon_lst = [self.epsilon]  # for plot
        self.lr = 0.5
        self.gamma = 0.9  # reward decay
        self.e_decay = 0.99995  # epsilon-decay schema
        self.q_table = np.zeros((48, len(self.all_actions)))  # Q-table
        # self.load_qtable()

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm.
        epsilon probs, choose the a that max q(s,a)
        1 - epsilon probs, choose a random action from self.all_actions
        """
        if np.random.uniform() > self.epsilon:  # epsilon probs, np.random.uniform() return a random float in the interval [0.0, 1.0)
            # choose the action with max q value
            action = np.argmax(self.q_table[observation, :])
        else:
            # choose a random action
            action = np.random.choice(self.all_actions)
        self.epsilon = self.epsilon * self.e_decay  # epsilon decay
        self.epsilon_lst.append(self.epsilon)
        return action

    def learn(self, state1, action1, reward, state2):
        """learn from experience"""
        print("I should learn! (ﾉ｀⊿´)ﾉ, epsilon: ", self.epsilon, "e_decay: ", self.e_decay, "lr: ", self.lr, "gamma: ", self.gamma)
        max_q_s2 = np.max(self.q_table[state2, :])
        self.q_table[state1][action1] = self.q_table[state1][action1] + self.lr * (
                    reward + self.gamma * max_q_s2 - self.q_table[state1][action1])
        return None

    def save_qtable(self):
        """You can add other functions as you wish."""
        # save the q_table to a file
        np.savetxt("qlearning/ql_q_table.txt", self.q_table)

        return None

    def save_epsilon(self):
        """You can add other functions as you wish."""
        # save the epsilon to a file
        np.savetxt("qlearning/ql_epsilon.txt", self.epsilon_lst)

        return None

    def load_qtable(self):
        """You can add other functions as you wish."""
        # load the q_table from a file
        self.q_table = np.loadtxt("qlearning/ql_q_table.txt")
        return None

    ##### END CODING HERE #####
