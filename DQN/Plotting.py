import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
import  pickle
from matplotlib import pyplot as plt

color = ['red', 'blue', 'green', 'yellow', 'cyan']


def plot_data(data_dict, episode_num, title, path='img/plot'):
    names = list(data_dict.keys())
    x = np.arange(len(data_dict[names[0]]))
    for i, name in enumerate(names):
        loss = np.array(data_dict[name])
        plt.plot(x, loss, '-', color=color[i], label=name, linewidth=0.5)
    plt.title(str(episode_num) + ' episodes ' + title )
    # plt.yscale("log")
    plt.xlabel('episode')
    plt.ylabel(title)
    plt.legend()
    # plt.savefig(path + title + str(episode_num)+'.png', dpi=300)
    plt.show()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_plot(data_dict, episode_num, title, path='img/plot'):
    names = list(data_dict.keys())
    for i, name in enumerate(names):
        x = np.arange(len(data_dict[name]))
        data = np.array(data_dict[name])
        data = smooth(data, 2)
        data[0:9] = -600
        plt.plot(x, data, '-', color=color[i], label=name, linewidth=1.4)
    plt.title(str(episode_num) + ' episodes ' + title + ' without smoothing')
    # plt.yscale("log")
    plt.xlabel('episode')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(path + title + str(episode_num) + '.png', dpi=300)
    plt.show()


def data_processing():
    DQN = pickle.load(open("data/DQN.txt", 'rb'))
    doubleDQN = pickle.load(open('data/doubleDQN.txt', 'rb'))
    duelingDQN = pickle.load(open('data/duelingDQN.txt', 'rb'))
    perDQN = pickle.load(open('data/perDQN.txt', 'rb'))
    ALL = pickle.load(open('data/ALL2.txt', 'rb'))
    reward = {'DQN': DQN['train'], 'double DQN': doubleDQN['train'],
            'dueling DQN': duelingDQN['train'], 'per DQN': perDQN['train'],
            'per Double Dueling DQN': ALL['train']}
    origin_reward = {'double DQN': doubleDQN['train'],
            # 'dueling DQN': duelingDQN['loss'], 'per DQN': perDQN['loss'],
            'per Double Dueling DQN': ALL['train']}
    # DQNlinear = pickle.load(open('data/DQN3.txt', 'rb'))
    # DQNnone = pickle.load(open('data/DQNnone.txt', 'rb'))
    # DQNexp = pickle.load(open('data/DQNexp.txt', 'rb'))
    # reward = {
    #           'DQN exponential reward': DQNexp['train']}
    #           # 'DQN linear reward': DQNlinear['train'],
    #           # 'DQN original reward': DQNnone['train']}
    smooth_plot(reward, 250, 'reward')


if __name__ == '__main__':
    data_processing()

