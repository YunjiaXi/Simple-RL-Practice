from tkinter import *
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import numpy as np
import Envs


class ModelFreeControl(object):
    def __init__(self, env, gamma=1, epsilon=0.1, alpha=0.02):
        self.env = env                                  # environment
        self.gamma = gamma                              # discount factor
        self.epsilon = epsilon                          # parameter for e-greedy
        self.alpha = alpha                              # learning rate
        self.states = env.getStates()                   # action space
        self.actions = env.getActions()                 # state space
        self.Q = {}                                     # action-value function
        for state in self.states:
            self.Q[state] = {}

    def initializeQ(self):
        # initialize Q to zero
        for state in self.states:
            for action in self.actions:
                self.Q[state][action] = 0
        # random initialization
        # for state in self.states:
        #     for action in self.actions:
        #         if state != self.env.end:
        #             self.Q[state][action] = random.random()
        #         else:
        #             self.Q[state][action] = 0

    def chooseActionByGreedy(self, state):
        max_one = max(self.Q[state].values())
        candidate = []
        for i in self.Q[state]:
            if self.Q[state][i] == max_one:
                candidate.append(i)
        return random.choice(candidate)

    def chooseActionByEGreedy(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.chooseActionByGreedy(state)

    def getPolicy(self):
        state = self.env.getStartState()
        is_end, path = False, defaultdict(list)
        while not is_end:
            action = self.chooseActionByGreedy(state)
            path[state].append(action)
            state, reward, is_end = self.env.step(state, action)
            print(action,state)
        return path

    def sarsa(self, theta=0.001):
        self.initializeQ()
        delta, k, deltas = 1, 0, []

        while delta > theta:
            delta, is_end = 0, False
            state = self.env.getStartState()
            action = self.chooseActionByEGreedy(state)
            # self.epsilon = 1/(k+2) # changeable epsilon
            while not is_end:
                next_state, reward, is_end = self.env.step(state, action)
                next_action = self.chooseActionByEGreedy(next_state)
                old_q = self.Q[state][action]
                self.Q[state][action] += self.alpha * (
                        reward + self.gamma * self.Q[next_state][next_action] - old_q)
                delta = max(delta, abs(self.Q[state][action] - old_q))
                state, action = next_state, next_action
            deltas.append(delta)
            k += 1
            # self.outputQ(k, 'Sarsa')
            print("episode ", k)
            if k > 15000: break
        self.plotDelta(deltas, k, 'Sarsa')
        self.plotQ(k, 'Sarsa')
        print('plotQDone')
        self.drawPath(self.getPolicy(), 'Sarsa')

    def lambdSarsa(self, episode_num=20000, lambd=0.9):
        self.initializeQ()
        delta, k, s, deltas = 1, 0, 0, []
        E = [{} for i in range(len(self.states))]

        # while delta > theta:
        while k < episode_num:
            for state in self.states:
                for action in self.actions:
                    E[state][action] = 0
            delta, is_end = 0, False
            state = self.env.getStartState()
            action = self.chooseActionByEGreedy(state)

            while not is_end:
                next_state, reward, is_end = self.env.step(state, action)
                next_action = self.chooseActionByEGreedy(next_state)
                old_q = self.Q[state][action]
                target = reward + self.gamma * self.Q[next_state][next_action] - old_q
                E[state][action] += 1
                for s in self.states:
                    for a in self.actions:
                        old_q = self.Q[s][a]
                        self.Q[s][a] += self.alpha * target * E[s][a]
                        E[s][a] = self.gamma * lambd * E[s][a]
                        delta = max(delta, self.Q[s][a] - old_q)
                state, action = next_state, next_action
            deltas.append(delta)
            k += 1
            # self.outputQ(k, 'Lambda Sarsa')
            print("episode ", k)

        self.plotDelta(deltas, k, 'Lambda Sarsa')
        self.plotQ(k, 'Lambda Sarsa')
        self.drawPath(self.getPolicy(), 'Lambda Sarsa')

    def Q_learning(self, theta=0.0001):
        self.initializeQ()
        delta, k, s, deltas = 1, 0, 0, []

        while delta > theta:
            delta, is_end = 0, False
            state = self.env.getStartState()
            while not is_end:
                action = self.chooseActionByEGreedy(state)
                next_state, reward, is_end = self.env.step(state, action)
                next_action = self.chooseActionByGreedy(state)
                old_q = self.Q[state][action]
                self.Q[state][action] += self.alpha * (
                        reward + self.gamma * self.Q[next_state][next_action] - old_q)
                delta = max(delta, abs(self.Q[state][action] - old_q))
                state = next_state
                s += 1
            deltas.append(delta)
            k += 1
            # self.outputQ(k, 'Q-learning')
            print("episode ", k)
        print(k, s/k, s)
        self.plotDelta(deltas, k, 'Q-Learning')
        self.plotQ(k, 'Q-Learning')
        self.drawPath(self.getPolicy(), 'Q-Learning')

    def outputQ(self, episode_num, method):
        print(method, episode_num, 'episode')
        for state in self.Q:
            print(state, self.Q[state])

    def plotQ(self, episode_num, method):
        h, w = len(self.actions), len(self.states) // 4
        plotPos = [411, 412, 413, 414]
        fig = plt.figure()

        for order in range(4):
            data = [[0 for i in range(w)] for i in range(h)]
            for i in range(h):
                for j in range(order * w, (order + 1) * w):
                    data[i][j - order * w] = self.Q[self.states[j]][self.actions[i]]

            ax = fig.add_subplot(plotPos[order])
            ax.set_yticks(range(h))
            ax.set_yticklabels(self.actions)
            ax.xaxis.set_visible(False)
            im = ax.imshow(data, aspect=0.6)
            for i in range(h):
                for j in range(order * w, (order + 1) * w):
                    text = ax.text(j - order * w, i, '%.3f' % data[i][j - order * w], ha="center", va="center", color="w", fontdict={'size': 5})
            plt.colorbar(im)

        ax.xaxis.set_visible(True)
        ax.set_xticks(range(w))
        plt.suptitle(method + ' ' + str(episode_num) + ' Episodes, Q')
        plt.savefig('img/plotValue'+method + str(episode_num)+'.png', dpi=200)
        plt.show()

    def plotDelta(self, deltas, episode_num, method):
        x = np.arange(len(deltas))
        plt.plot(x, deltas, '-', color="darkcyan", label=str(episode_num) + ' episode', linewidth=0.5)
        plt.title(method + ' delta')
        plt.yscale("log")
        plt.xlabel('episode')
        plt.ylabel('delta')
        plt.legend()
        plt.savefig('img/plotDelta'+method + str(episode_num)+'.png', dpi=300)
        plt.show()

    def drawPath(self, path, name=None):
        print(path)
        app = Tk()
        app.title(name)
        canvas = Canvas(app, bg='white', width=800, height=400)
        canvas.pack()

        # draw arrow in different directions
        def drawArrow(x, y, length, direction):
            direct_diff = {'up': (0, -length), 'down': (0, length), 'left': (-length, 0), 'right': (length, 0)}
            end_x, end_y = x + direct_diff[direction][0], y + direct_diff[direction][1]
            canvas.create_line(x, y, end_x, end_y)
            if direction == 'up':
                canvas.create_line(end_x, end_y, end_x + 4, end_y + 4)
                canvas.create_line(end_x, end_y, end_x - 4, end_y + 4)
            elif direction == 'down':
                canvas.create_line(end_x, end_y, end_x + 4, end_y - 4)
                canvas.create_line(end_x, end_y, end_x - 4, end_y - 4)
            elif direction == 'left':
                canvas.create_line(end_x, end_y, end_x + 4, end_y - 4)
                canvas.create_line(end_x, end_y, end_x + 4, end_y + 4)
            else:
                canvas.create_line(end_x, end_y, end_x - 4, end_y - 4)
                canvas.create_line(end_x, end_y, end_x - 4, end_y + 4)

        # draw policy
        begin_x, begin_y, edge, arrow_len = 100, 50, 50, 20
        h, w = self.env.shape
        color = ['white', '#708090', '#7AC5CD', '#66CDAA']
        for i in range(h):
            for j in range(w):
                state = i * w + j
                x, y = begin_x + j * edge, begin_y + i * edge
                canvas.create_rectangle(x, y, x + edge, y + edge, fill=color[self.env.grid[(i, j)]])
                for action in path[state]:
                    drawArrow(x + edge // 2, y + edge // 2, arrow_len, action)
        mainloop()


def main():
    env = Envs.CliffWalking()
    model = ModelFreeControl(env)
    # model.sarsa()
    model.Q_learning()
    # model.lambdSarsa()


if __name__ == '__main__':
    main()