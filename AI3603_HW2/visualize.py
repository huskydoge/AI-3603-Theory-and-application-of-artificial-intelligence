from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd
import os.path as osp
import os

def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1)
    anim.save('./breakout_result.gif', writer='ffmpeg', fps=30)


def plot_episode_reward(episode_reward_lst, episode=999):
    """plot the episode reward"""
    plt.figure()
    plt.plot(np.arange(len(episode_reward_lst[episode])), episode_reward_lst[episode])
    plt.xlabel('iteration')
    plt.ylabel('episode reward')
    plt.savefig('episode_reward.svg')
    plt.show()

    return None


def plot_epsilon(epsilon_lst,save_path="./sarsa"):
    """plot the epsilon"""
    plt.figure()
    plt.plot(np.arange(len(epsilon_lst)), epsilon_lst)
    plt.xlabel('iteration')
    plt.ylabel('epsilon')
    plt.tight_layout()
    plt.savefig(osp.join(save_path,'epsilon.svg'))
    plt.show()
    return None


def plot_episode_state_map(episode_state_map, episode=999,save_path="./sarsa"):
    plt.figure()
    map = episode_state_map[episode].reshape(4, 12)

    # Calculate the sum of values for all episodes
    sum_map = np.sum(episode_state_map[:episode + 1], axis=0).reshape(4, 12)

    # Normalize the values for color mapping
    max_value = np.max(sum_map)
    min_value = np.min(sum_map)

    # Use the 'hot' colormap and adjust colors based on normalized values
    plt.imshow(sum_map, cmap='hot', interpolation='nearest', norm=plt.Normalize(vmin=min_value, vmax=max_value))

    # Show the numbers in the map with smaller size for larger values and darker colors
    for i in range(4):
        for j in range(12):
            value = sum_map[i, j]
            text_color = 'black' if value > max_value * 0.5 else 'white'  # Adjust text color based on value
            plt.text(j, i, int(value), ha="center", va="center", color=text_color, fontsize=6)

    # Color bar, smaller, fit the image
    plt.colorbar(shrink=0.3)
    plt.tight_layout()
    plt.savefig(osp.join(save_path,'episode_state_map.svg'))
    plt.show()


def plot_reward_per_episode(episode_reward_lst,save_path="./sarsa"):
    """plot the reward per episode"""
    plt.figure()
    # set the x title and y title font size to be larger and bolder
    plt.rcParams['axes.titlesize'] = 20
    lst = [episode_reward_lst[i][-1] for i in range(1000)]
    mean_lst = [np.mean(lst[:i]) for i in range(1, len(lst) + 1)]
    # plot reward per episode, use a lighter color
    plt.plot(np.arange(len(lst)), lst, label='reward per episode', alpha=0.3)
    # smoothing the plot
    plt.plot(np.arange(len(lst)), pd.Series(lst).rolling(10).mean(), label='smoothed reward per episode')
    # mean reward, each point is the mean of all the rewards before it
    plt.plot(np.arange(len(lst)), mean_lst, label='mean reward per episode')

    # annotate the max value in the plot with a red dot and a axline
    plt.plot(np.argmax(lst), np.max(lst), 'r.', markersize=10,label='max reward')
    plt.axvline(np.argmax(lst), color='r', linestyle='--')
    # annotate its x pos in x axis with a tick
    plt.annotate(f"{np.max(lst)}", xy=(np.argmax(lst), np.max(lst)), xytext=(np.argmax(lst) + 10, np.max(lst) + 15),
                 )
    plt.xticks([0,200,400,600,800,1000,np.argmax(lst)], ['',200,400,600,800,1000,f'{np.argmax(lst)}'])
    # plt.xticks([len(lst) - 1], [str(len(lst))])
    # legend, blue is the reward per episode, orange is the smoothed reward per episode
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.savefig(osp.join(save_path,'reward_per_episode.svg'))
    # tight layout
    plt.tight_layout()
    plt.show()
    return None


if __name__ == "__main__":


    # sarsa
    episode_reward_lst = np.load(
        "/Users/husky/Three-Year-Aut/AI-theory/AI3603_HW2/sarsa/episode_reward_lst.npy",
        allow_pickle=True)
    # to dict
    episode_reward_lst = episode_reward_lst.item()
    epsilon = np.loadtxt("/Users/husky/Three-Year-Aut/AI-theory/AI3603_HW2/sarsa/sarsa_epsilon.txt")

    plot_reward_per_episode(episode_reward_lst)
    plot_epsilon(epsilon)

    state_map = np.load("/Users/husky/Three-Year-Aut/AI-theory/AI3603_HW2/sarsa/episode_state_map.npy")
    plot_episode_state_map(state_map, episode=999)
    #
    # qlearning
    episode_reward_lst = np.load(
        "/Users/husky/Three-Year-Aut/AI-theory/AI3603_HW2/qlearning/ql_episode_reward_lst.npy",
        allow_pickle=True)
    # to dict
    episode_reward_lst = episode_reward_lst.item()
    epsilon = np.loadtxt("/Users/husky/Three-Year-Aut/AI-theory/AI3603_HW2/qlearning/ql_epsilon.txt")

    plot_reward_per_episode(episode_reward_lst,save_path="./qlearning")
    plot_epsilon(epsilon,save_path="./qlearning")

    state_map = np.load("/Users/husky/Three-Year-Aut/AI-theory/AI3603_HW2/qlearning/ql_episode_state_map.npy")
    plot_episode_state_map(state_map, episode=999,save_path="./qlearning")

