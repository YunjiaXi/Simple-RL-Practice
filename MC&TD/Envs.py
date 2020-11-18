import random


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
