import random
import Envs
import matplotlib.pyplot as plt
import numpy as np


class ModelFree(object):
    def __init__(self, environment, policy=None, gamma=1):
        self.environment = environment
        self.states = environment.getStates()
        self.state_num = len(self.states)
        self.policy = policy
        self.gamma = gamma
        if not policy:
            self.policy = {state: environment.getActions() for state in self.states}
            for state in environment.getTerminal():
                self.policy[state] = []

    def setPolicy(self, policy):
        self.policy = policy

    def outputValue(self, values, k, method):
        print(method + " Evaluation", k, "times, Value Function:")
        grid_len = self.environment.grid_len
        grid_matrix = [[0 for i in range(grid_len)] for i in range(grid_len)]
        for i in range(grid_len):
            print('\n' + '|' + ('-' * 11 + '|') + ('-' * 10 + '|') * 5)
            print("|", end=' ')
            for j in range(grid_len):
                state = i * grid_len + j
                print(('%.4f' % values[state]).center(10), end='|')
                grid_matrix[i][j] = values[state]
        print('\n' + '|' + ('-' * 11 + '|') + ('-' * 10 + '|') * 5)
        print('\n')

    def plotValues(self, values, episode_num, method):
        grid_len = self.environment.grid_len
        data = [[0 for i in range(grid_len)] for i in range(grid_len)]
        for i in range(grid_len):
            for j in range(grid_len):
                data[i][j] = values[i * grid_len + j]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_yticks(range(grid_len))
        ax.set_xticks(range(grid_len))
        im = ax.imshow(data)
        for i in range(grid_len):
            for j in range(grid_len):
                text = ax.text(j, i, '%.2f' % data[i][j], ha="center", va="center", color="w")

        plt.colorbar(im)
        plt.title(method + ' ' + str(episode_num) + ' Episodes, Values')
        # plt.savefig('img/plotValue'+method + str(episode_num)+'.png', dpi=200)
        plt.show()

    def plotDelta(self, deltas, episode_num, method):
        x = np.arange(len(deltas))
        plt.plot(x, deltas, '-', color="darkcyan", label=str(episode_num) + ' episode', linewidth=0.5)
        plt.title(method + ' Prediction delta')
        plt.yscale("log")
        plt.xlabel('episode')
        plt.ylabel('delta')
        plt.legend()
        # plt.savefig('img/plotDelta'+method + str(episode_num)+'.png', dpi=300)
        plt.show()

    def plotMultiDelta(self, deltas, episode_num, method):
        x = np.arange(len(deltas[0]))
        plt.plot(x, deltas[0], '-', color="forestgreen", label='alpha=0.1', linewidth=0.5)
        plt.plot(x, deltas[1], '-', color="darkcyan", label='alpha=0.01', linewidth=0.5)
        plt.plot(x, deltas[2], '-', color="royalblue", label='alpha=0.001', linewidth=0.5)
        plt.title(method + ' Prediction delta')
        plt.yscale("log")
        plt.xlabel('episode')
        plt.ylabel('delta')
        plt.legend()
        # plt.savefig('img/plotMultiDelta'+method + str(episode_num)+'.png', dpi=300)
        plt.show()


class MC(ModelFree):
    def computeEpisodeReturn(self, ep_states, ep_reward):
        ep_len = len(ep_states)
        ep_returns = [0 for i in range(ep_len)]
        ep_returns[-1] = ep_reward[-1]
        for i in range(ep_len-2, -1, -1):
            ep_returns[i] = ep_reward[i] + self.gamma * ep_returns[i + 1]
        return ep_returns

    def _MCPolicyEval(self, episode_num, visit_method, use_alpha=False, alpha=0.01):
        values = {state: 0 for state in self.states}     # initialize value for each state
        state_cnt = {state: 0 for state in self.states}  # initialize count for each state
        k, deltas = 0, []

        while k < episode_num:
            delta = 0
            ep_states, ep_actions, ep_reward = self.environment.generateEpisode(self.policy)
            ep_returns = self.computeEpisodeReturn(ep_states, ep_reward)
            # record whether the state is visited in this episode
            first_appear = {state: True for state in self.states}
            for i, state in enumerate(ep_states):
                # if state isn't visited in this episode before or use every-visit, update state
                if first_appear[state] or visit_method == 'every':
                    state_cnt[state] += 1
                    old_value = values[state]
                    if not use_alpha:
                        values[state] += (ep_returns[i] - old_value) / state_cnt[state]
                    else:
                        values[state] += (ep_returns[i] - old_value) * alpha
                    first_appear[state] = False
                    delta = max(delta, abs(values[state] - old_value))
            deltas.append(delta)
            k += 1
            if k % 100 == 0:
                self.outputValue(values, k, visit_method + '-visit MC')
            # if k in [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000]:
            #     self.plotDelta(deltas, k, visit_method + '-visit MC')
            #     self.plotValues(values, k, visit_method + '-visit MC')
        self.plotDelta(deltas, k, visit_method + '-visit MC')
        self.plotValues(values, k, visit_method + '-visit MC')

    def firstVisitMCPolicyEval(self, episode_num, use_alpha=False, alpha=0.01):
        self._MCPolicyEval(episode_num, 'first', use_alpha, alpha)

    def everyVisitMCPolicyEval(self, episode_num, use_alpha=False, alpha=0.01):
        self._MCPolicyEval(episode_num, 'every', use_alpha, alpha)


class TD(ModelFree):
    def TD0PolicyEval(self, episode_num, alpha=0.1):
        values = {state: 0 for state in self.states}
        deltas = []
        k = 0

        while k < episode_num:
            delta, is_end = 0, False
            cur_state = self.environment.generateInitialState()
            while not is_end:
                cur_action = random.choice(self.policy[cur_state])
                cur_reward, next_state, is_end = self.environment.getNextState(cur_state, cur_action)
                old_value = values[cur_state]
                values[cur_state] += alpha * (cur_reward + self.gamma * values[next_state] - old_value)
                delta = max(delta, abs(values[cur_state] - old_value))
                cur_state = next_state
            deltas.append(delta)
            k += 1
            if k % 100 == 0:
                self.outputValue(values, k, 'TD(0)')
                # self.plotValues(values, k, 'TD(0)')
            # if k in [20, 30, 40, 50, 601200, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000]:
            #     self.plotDelta(deltas, k, 'TD(0)')
            #     self.plotValues(values, k, 'TD(0)')
        self.plotValues(values, k, 'TD(0)')
        self.plotDelta(deltas, k, 'TD(0)')
        return deltas


def main():
    episode_num = 1e4
    optimal_policy = [['e'], [], ['w'], ['w'], ['w'], ['w'],
['n', 'e'], ['n'], ['n', 'w'], ['n', 'w'], ['n', 'w'], ['s'],
['n', 'e'], ['n'], ['n', 'w'], ['n', 'w'], ['s', 'e'], ['s'],
['n', 'e'], ['n'], ['n', 'w'], ['s', 'e'], ['s', 'e'], ['s'],
['n', 'e'], ['n'], ['s', 'e'], ['s', 'e'], ['s', 'e'], ['s'],
['e'], ['e'], ['e'], ['e'], ['e'], []]
    policy = {i: optimal_policy[i] for i in range(36)}

    grid_world = Envs.GridWorld(6, [1, 35])
    # optimal policy
    # MC_model = MC(grid_world, policy)
    # MC_model.firstVisitMCPolicyEval(episode_num)
    # MC_model.everyVisitMCPolicyEval(episode_num)
    # TD_model = TD(grid_world, policy)
    # TD_model.TD0PolicyEval(episode_num, 1)

    # uniform random policy
    MC_model = MC(grid_world)
    MC_model.firstVisitMCPolicyEval(episode_num)
    # MC_model.everyVisitMCPolicyEval(episode_num)
    # TD_model = TD(grid_world)
    # TD_model.TD0PolicyEval(episode_num)

    # deltas = [TD_model.TD0PolicyEval(episode_num, 0.1), TD_model.TD0PolicyEval(episode_num, 0.01),
    #           TD_model.TD0PolicyEval(episode_num, 0.001)]
    # TD_model.plotMultiDelta(deltas, episode_num, "TD(0)")


if __name__ == '__main__':
    main()
