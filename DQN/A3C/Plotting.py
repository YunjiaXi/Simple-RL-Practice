import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle
from matplotlib import pyplot as plt
import seaborn

color = ['red', 'blue', 'green', 'yellow', 'cyan']


def plot_data(data_dict, episode_num, title, path='img/plot'):
    names = list(data_dict.keys())
    x = np.arange(len(data_dict[names[0]]))
    for i, name in enumerate(names):
        loss = np.array(data_dict[name])
        plt.plot(x, loss, '-', color=color[i], label=name, linewidth=0.5)
    plt.title(str(episode_num) + ' episodes ' + title)
    # plt.yscale("log")
    plt.xlabel('episode')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(path + title + str(episode_num)+'.png', dpi=300)
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


def plot_mean_data(data_dict, episode_num, title='5 step updating Average Reward', path='img/plot'):
    names = list(data_dict.keys())
    fig = plt.figure()
    seaborn.set(style="darkgrid", font_scale=1.5)
    seaborn.set_context("notebook", rc={"lines.linewidth": 1})
    for i, name in enumerate(names):
        for j, x in enumerate(data_dict[name]):
            new_data = [sum(x[:9])/9]
            for k in range(9, 2999, 10):
                new_data.append(sum(x[k:k+10])/10)
            data_dict[name][j] = new_data
            # data_dict[name][j] = x[:2999]
        # data = np.array(data_dict[name])
        # mean = np.mean(data, axis=0)
        # std = np.std(data, axis=0)
        # half_reward_std = std / 2.0
        # lower = [x - y for x, y in zip(mean, half_reward_std)]
        # upper = [x + y for x, y in zip(mean, half_reward_std)]
        # x = np.arange(len(mean))
        # plt.plot(x, mean, color=color[i])
        # plt.fill_between(x, lower, upper, color=color[i], alpha=0.2)
        # plt.grid()
        x = np.arange(0, 2999, 10)
        seaborn.tsplot(time=x, data=data_dict[name], color=color[i], condition=name)
    plt.title('A3C ' + str(episode_num) + ' episodes ' + title)
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.savefig(path + title + str(episode_num) + '.png', dpi=300)
    plt.show()


def DQN_data_processing():
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


def A3C_data_processing():
    data1 = pickle.load(open("data/SharedAdam_3000ep_400sp_5gst_worker4.txt", 'rb'))
    data2 = pickle.load(open("data/SharedAdam_3000ep_400sp_5gst_4.txt", 'rb'))
    # data3 = pickle.load(open("data/RMSprop_3000ep_400sp_5gst.txt", 'rb'))
    # data4 = pickle.load(open("data/SharedRMSprop_3000ep_400sp_5gst.txt", 'rb'))
    data_dict = {'4 workers': data1, '8 workers': data2}
    plot_mean_data(data_dict, 3000)


def plot_one_ac(name="data/SharedAdam_3000ep_400sp_5gst.txt"):
    data1 = pickle.load(open(name, 'rb'))
    data_dict = {'5 step': data1}
    plot_mean_data(data_dict, 3000)


if __name__ == '__main__':
    A3C_data_processing()

