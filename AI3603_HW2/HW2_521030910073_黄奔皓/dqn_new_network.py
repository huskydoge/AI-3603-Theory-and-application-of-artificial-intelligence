# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from gym import wrappers
def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, # 可调参数
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.5,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=2,
        help="the frequency of training")
    parser.add_argument("--test", type=bool, default=False,
        help="test the results")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = wrappers.RecordVideo(env, './videos/')
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """comments: Q network used for DQN, which could provide the q value of current obs"""
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 256),  # Increase hidden layer size
            nn.ReLU(),
            nn.Linear(256, 128),  # Increase hidden layer size
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """comments:  linearly decay the epsilon from start_e to end_e"""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    
    """parse the arguments"""
    print("start training")
    args = parse_args()
    isTest = args.test
    model_path = "/Users/husky/Three-Year-Aut/AI-theory/AI3603_HW2/models/LunarLander-v2__dqn__42__1699249744.pth"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    """comments: set the random seed so that the results could be reproduced"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    """comments: get the environment we use for training"""
    envs = make_env(args.env_id, args.seed)
    """comments: init two networks, one for training, one for target; init the optimizer"""
    if not isTest:
        q_network = QNetwork(envs).to(device)
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
        target_network = QNetwork(envs).to(device)
        target_network.load_state_dict(q_network.state_dict())
    else:
        q_network = QNetwork(envs).to(device)
        q_network.load_state_dict(torch.load(model_path))
        target_network = QNetwork(envs).to(device)
        target_network.load_state_dict(q_network.state_dict())

    """comments: rb is used to store the experience"""
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    if isTest:
        obs = envs.reset()
        for global_step in range(10, args.total_timesteps):

            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()

            """comments: step the env and get the next_obs, rewards, dones, infos,
            here infos include the episodic_return and episodic_length, """
            next_obs, rewards, dones, infos = envs.step(actions)
            envs.render() # close render during training

            """comments: learn from experience, store the experience in the replay buffer"""
            rb.add(obs, next_obs, actions, rewards, dones, infos)

            """comments: step to the next observation"""
            obs = next_obs if not dones else envs.reset()

    else:
        """comments: init the observation"""
        obs = envs.reset()
        for global_step in range(args.total_timesteps):

            """comments: linearly decay the epsilon from start_e to end_e"""
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

            """comments: epsilon-greedy algorithm"""
            if random.random() < epsilon:
                actions = envs.action_space.sample()
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=0).cpu().numpy()

            """comments: step the env and get the next_obs, rewards, dones, infos,
            here infos include the episodic_return and episodic_length, """
            next_obs, rewards, dones, infos = envs.step(actions)
            # envs.render() # close render during training

            if dones:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

            """comments: learn from experience, store the experience in the replay buffer"""
            rb.add(obs, next_obs, actions, rewards, dones, infos)

            """comments: step to the next observation"""
            obs = next_obs if not dones else envs.reset()

            if global_step > args.learning_starts and global_step % args.train_frequency == 0:

                """comments: choose a batch of data from the replay buffer"""
                data = rb.sample(args.batch_size)

                """comments: evaluate the q value of the current network, and contrast it with the target network"""
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                """comments: for visualizationm, we log the loss and q values"""
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

                """comments: optimize the network"""
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                """comments:update target network, every args.target_network_frequency steps"""
                if global_step % args.target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())
        if not isTest:
            torch.save(q_network.state_dict(), f"models/{run_name}.pth")
        """close the env and tensorboard logger"""
    envs.close()
    writer.close()