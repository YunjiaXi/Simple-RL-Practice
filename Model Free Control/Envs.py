import random
import numpy as np

class GridWorld(object):
    def __init__(self, grid_len, terminal, gamma=1):
        self.grid_len = grid_len  # edge length of grid word
        self.states_num = grid_len ** 2  # number of states
        self.states = [i for i in range(self.states_num)]  # all states
        self.terminal = terminal  # all terminal states
        self.reward = -1  # reward
        self.actions = ['n', 's', 'w', 'e']  # all actions
        # how action lead to state changing
        self.act_trans = {'n': -grid_len, 's': grid_len, 'w': -1, 'e': 1}

    def getStates(self):
        return self.states

    def getTerminal(self):
        return self.terminal

    def getActions(self):
        return self.actions.copy()

    def generateInitialState(self):
        init_state = random.choice(self.states)
        while init_state in self.terminal:
            init_state = random.choice(self.states)
        return init_state

    def getNextState(self, state, action):
        next_state = state + self.act_trans[action]
        # return to current state if next state is out of grid
        if next_state < 0 or next_state >= self.states_num or (
                next_state % self.grid_len != state % self.grid_len and
                next_state // self.grid_len != state // self.grid_len):
            next_state = state
        is_end = next_state in self.terminal
        return self.reward, next_state, is_end

    def generateEpisode(self, policy):
        ep_states, ep_actions, ep_rewards = [], [], []
        cur_state = self.generateInitialState()
        is_end = False
        while not is_end:
            ep_states.append(cur_state)
            cur_action = random.choice(policy[cur_state])
            ep_actions.append(cur_action)
            cur_reward, cur_state, is_end = self.getNextState(cur_state, cur_action)
            ep_rewards.append(cur_reward)
        return ep_states, ep_actions, ep_rewards


class CliffWalking(object):
    def __init__(self, shape=(4, 12), start=(3, 0), end=(3, 11)):
        self.shape = shape
        self.start = self._pos_to_state(start)
        self.end = self._pos_to_state(end)
        self.actions = ['up', 'down', 'right', 'left']
        self.states = [i for i in range(self.shape[0] * self.shape[1])]

        self.grid = np.zeros(self.shape, dtype=np.int32)
        for i in range(1, self.shape[1]-1):
            self.grid[3][i] = 1  # cliff
        self.grid[start] = 2     # start
        self.grid[end] = 3       # end
        # print(self.grid)

        self.transition = {}
        for state in self.states:
            self.transition[state] = {}
            position = self._state_to_pos(state)
            if self.grid[tuple(position)] in [0, 2]:
                self.transition[state]['up'] = self._calculate_transition(position, [-1, 0])
                self.transition[state]['down'] = self._calculate_transition(position, [1, 0])
                self.transition[state]['left'] = self._calculate_transition(position, [0, -1])
                self.transition[state]['right'] = self._calculate_transition(position, [0, 1])
        # for i in self.states:
        #     print(i, self.transition[i])

    def _pos_to_state(self, pos):
        return pos[0] * self.shape[1] + pos[1]

    def _state_to_pos(self, state):
        return [state // self.shape[1], state % self.shape[1]]

    def _calculate_transition(self, position, move):
        new_pos = np.array(position) + np.array(move)
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.shape[0] or new_pos[1] >= self.shape[1]:
            new_pos = position
        new_state = self._pos_to_state(new_pos)
        reward = -1
        if self.grid[tuple(new_pos)] == 1:
            reward = -100
            new_state = self.start
        is_end = new_state == self.end
        return [new_state, reward, is_end]

    def getStartState(self):
        return self.start

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def getTerminal(self):
        return self.end

    def step(self, state, action):
        return self.transition[state][action]
