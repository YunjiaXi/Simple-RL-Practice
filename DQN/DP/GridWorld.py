from tkinter import *
import random


class GridWord(object):
    def __init__(self, grid_len, terminal, gamma=1):
        self.grid_len = grid_len                                              # edge length of grid word
        self.states_num = grid_len ** 2                                       # number of states
        self.states = [i for i in range(self.states_num)]                     # all states
        self.values = [0 for i in range(self.states_num)]                     # values of states
        self.terminal = terminal                                              # all terminal states
        self.gamma = gamma                                                    # discount factor
        self.reward = -1                                                      # reward
        self.actions = ['n', 's', 'w', 'e']                                   # all actions
        self.act_trans = {'n': -grid_len, 's': grid_len, 'w': -1, 'e': 1}     # how action lead to state changing
        self.policy = [['n', 's', 'w', 'e'] for i in range(self.states_num)]  # policy at each state, initially uniform
        for x in self.terminal:                                               # terminate at terminal states
            self.policy[x] = []

    def getNextState(self, state, action):
        next_state = state + self.act_trans[action]
        # return to current state if next state is out of grid
        if next_state < 0 or next_state >= self.states_num or (
                next_state % self.grid_len != state % self.grid_len and next_state // self.grid_len != state // self.grid_len):
            next_state = state
        return next_state

    def policyEvaluation(self):
        theta = 0.01
        delta = 1
        k = 0
        self.outputValue(k)
        while delta > theta:
            delta = 0
            # synchronous backups
            new_values = [self.values[i] if i in self.terminal else 0 for i in range(self.states_num)]
            for state in self.states:
                if self.policy[state]:
                    trans_prop = 1 / (len(self.policy[state]))
                for action in self.policy[state]:
                    next_state = self.getNextState(state, action)
                    new_values[state] += (self.reward + self.gamma * self.values[next_state]) * trans_prop
                delta = max(delta, abs(new_values[state] - self.values[state]))

            # update value function
            self.values = new_values
            k += 1
            self.outputValue(k)

    def policyIteration(self):
        stable = False
        k = 0
        self.outputPolicy(k)

        while not stable:
            k += 1
            print('----------------------------------- Policy Iteration', k, 'times-----------------------------------')
            self.policyEvaluation()

            stable = True
            for state in self.states:
                old_policy = self.policy[state]
                if state not in self.terminal:
                    # find best policy
                    max_value, policy = -9e10, []
                    for action in self.actions:
                        next_state = self.getNextState(state, action)
                        new_value = self.reward + self.gamma * self.values[next_state]
                        if new_value > max_value:
                            max_value = new_value
                            policy = [action]
                        elif abs(new_value - max_value) < 1e-9:
                            policy.append(action)
                    # update policy
                    self.policy[state] = policy

                    if set(old_policy) != set(policy):
                        stable = False

            self.outputPolicy(k)

    def valueIteration(self):
        theta = 0.01
        delta = 1
        k = 0

        while delta > theta:
            delta = 0
            new_values = [0 for i in range(self.states_num)]
            for state in self.states:
                if state not in self.terminal:
                    max_value, policy = -9e9, []
                    for action in self.actions:
                        next_state = self.getNextState(state, action)
                        new_value = self.reward + self.gamma * self.values[next_state]
                        if new_value > max_value:
                            max_value = new_value
                            policy = [action]
                        elif abs(new_value - max_value) < 1e-9:
                            policy.append(action)
                    new_values[state] = max_value
                    delta = max(delta, abs(max_value - self.values[state]))
                    self.policy[state] = policy
            self.values = new_values
            k += 1
            self.outputValue(k)
            self.outputPolicy(k)




    # print value function at every step
    def outputValue(self, k):
        print("Policy Evaluation", k, "times, Value Function:")
        for i in range(self.grid_len):
            print('\n' + '|' + ('-'*11 + '|') + ('-'*10 + '|') * 5)
            print("|", end=' ')
            for j in range(self.grid_len):
                state = i * self.grid_len + j
                print(('%.4f' % self.values[state]).center(10), end='|')
        print('\n' + '|' + ('-'*11 + '|') + ('-'*10 + '|') * 5)
        print('\n')

    def outputPolicy(self, k):
        print("Policy Iteration", k, "times, Optimal Policy")
        # window and canvas
        app = Tk()
        app.title('Policy Iteration ' + str(k) + ' times, Optimal Policy:')
        canvas = Canvas(app, bg='white', width=800, height=800)
        canvas.pack()

        # draw arrow in different directions
        def drawArrow(x, y, length, direction):
            direct_diff = {'n': (0, -length), 's': (0, length), 'w': (-length, 0), 'e': (length, 0)}
            end_x, end_y = x + direct_diff[direction][0], y + direct_diff[direction][1]
            canvas.create_line(x, y, end_x, end_y)
            if direction == 'n':
                canvas.create_line(end_x, end_y, end_x + 4, end_y + 4)
                canvas.create_line(end_x, end_y, end_x - 4, end_y + 4)
            elif direction == 's':
                canvas.create_line(end_x, end_y, end_x + 4, end_y - 4)
                canvas.create_line(end_x, end_y, end_x - 4, end_y - 4)
            elif direction == 'w':
                canvas.create_line(end_x, end_y, end_x + 4, end_y - 4)
                canvas.create_line(end_x, end_y, end_x + 4, end_y + 4)
            else:
                canvas.create_line(end_x, end_y, end_x - 4, end_y - 4)
                canvas.create_line(end_x, end_y, end_x - 4, end_y + 4)

        # draw policy
        begin_x, begin_y, edge, arrow_len = 100, 50, 80, 30
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                state = i * self.grid_len + j
                print(self.policy[state], end=' ')
                x, y = begin_x + j * edge, begin_y + i * edge
                canvas.create_rectangle(x, y, x + edge, y + edge)
                for action in self.policy[state]:
                    drawArrow(x + edge // 2, y + edge // 2, arrow_len, action)
            print()
        print()
        mainloop()


def main():
    grid = GridWord(6, [1, 35])
    grid.policyIteration()
    # grid.valueIteration()


if __name__ == "__main__":
    main()
