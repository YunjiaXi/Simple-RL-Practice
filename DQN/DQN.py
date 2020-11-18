import gym
import torch
import torch.nn as nn
import math
import pickle
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
from memory import ReplayMemory
from memory import PERMemory
from Plotting import plot_data

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim, out_dim)
        # weight init
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = self.linear3(x)
        return x


class DuelingNet(nn.Module):
    def __init__(self, state_dim, action_dim, hid_dim):
        super(DuelingNet, self).__init__()
        self.state_space = state_dim
        self.fc1 = nn.Linear(self.state_space, hid_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.action_space = action_dim
        self.fc_z_v = nn.Linear(hid_dim, 1)
        self.fc_z_v.weight.data.normal_(0, 0.1)
        self.fc_z_a = nn.Linear(hid_dim, self.action_space)
        self.fc_z_a.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        v, a = self.fc_z_v(x), self.fc_z_a(x)  # Calculate value and advantage streams
        a_mean = torch.stack(a.chunk(self.action_space, 1), 1).mean(1)
        x = v.repeat(1, self.action_space) + a - a_mean.repeat(1, self.action_space)  # Combine streams
        return x


class DQN(object):
    def __init__(self, state_dim, action_dim, gamma=1, batch_size=128, lr=1e-3,
                 epsilon=0.1, memory_size=1000, target_update=200, hidden_dim=100,
                 use_per=False, use_double=False, use_dueling=False):
        self.action_num = action_dim                    # action space
        self.target_update = target_update              # frequency of target net updating
        self.gamma = gamma                              # discount factor
        self.epsilon = epsilon                          # epsilon-greedy parameter
        self.batch_size = batch_size                    # batch size
        self.learning_rate = lr                         # learning late
        self.use_per = use_per                          # whether use priority experience replay
        self.use_double = use_double                    # whether use Double-DQN
        self.train_step_cnt = 0                         # count the training step
        self.replay_buffer = ReplayMemory(memory_size) if not use_per else PERMemory(memory_size)
        # initialize net
        if use_dueling:                                 # whether use Dueling-DQN
            self.eval_net = DuelingNet(state_dim, action_dim, hidden_dim)
            self.target_net = DuelingNet(state_dim, action_dim, hidden_dim)
        else:
            self.eval_net = Net(state_dim, action_dim, hidden_dim)
            self.target_net = Net(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()                   # optimizer & loss function

    def choose_action(self, state):
        # epsilon-greedy, used for train
        state = torch.unsqueeze(torch.tensor(state).float(), 0)
        if np.random.uniform() > self.epsilon - self.train_step_cnt*0.0002:
            actions_value = self.eval_net.forward(state)
            action = torch.argmax(actions_value, dim=1).numpy()[0]
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def choose_greedy_action(self, state):
        # greedy, used for test
        state = torch.unsqueeze(torch.tensor(state).float(), 0)
        actions_value = self.eval_net.forward(state)
        action = torch.argmax(actions_value, dim=1).numpy()[0]
        return action

    def get_td_error(self, packed):
        # calculate td_error as priority
        state = torch.unsqueeze(torch.tensor(packed.state).float(), 0)
        next_state = torch.unsqueeze(torch.tensor(packed.next_state).float(), 0)
        action = torch.unsqueeze(torch.tensor([packed.action]), 0)
        q_eval = self.eval_net.forward(state).gather(1, action)[0]
        q_next = self.target_net.forward(next_state)
        q_target = packed.reward + self.gamma * q_next.max(1)[0] * (1 - packed.done)
        td_error = abs((q_target - q_eval).item())
        return td_error

    def add_to_replay_buffer(self, td_error, packed):
        if self.use_per:
            self.replay_buffer.push(td_error, packed)
        else:
            self.replay_buffer.push(packed)

    def update(self):
        # update target net
        if self.train_step_cnt % self.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.train_step_cnt += 1

        # get batch data
        batch = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*batch))
        cur_states = torch.tensor(batch.state).float()
        actions = torch.unsqueeze(torch.tensor(batch.action), 1)
        rewards = torch.unsqueeze(torch.tensor(batch.reward), 1)
        next_states = torch.tensor(batch.next_state).float()
        done = torch.unsqueeze(torch.tensor(batch.done).float(), 1)

        # calculate loss
        q_eval = self.eval_net(cur_states)
        qa_eval = q_eval.gather(1, actions)             # shape (batch, action_dim)
        q_next = self.target_net(next_states)           # don't calculate the gradient of q_next
        if self.use_double:                             # whether use double DQN
            qa_next = q_next.gather(1, torch.unsqueeze(torch.argmax(q_eval, dim=1), 1)).detach()
        else:
            qa_next = torch.unsqueeze(q_next.max(1)[0], 1).detach()
        q_target = rewards + self.gamma * qa_next * (1 - done)
        loss = self.loss_func(qa_eval, q_target)

        # update eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path='agent.pth'):
        torch.save(self.eval_net, path)

    def load(self, path):
        torch.load(self.eval_net, path)
        torch.load(self.target_net, path)


def test(agent, env, render=False):
    total_reward = 0
    state = env.reset()
    while True:
        if render:
            env.render()
        action = agent.choose_greedy_action(state)  # direct action for test
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print("test reward ", total_reward)
    return total_reward


def train(agent, env, episode_num):
    memory_cnt = 0
    loss, test_rewards, train_rewards, mod_rewards = [], [], [], []
    for i in range(episode_num):
        total_reward, mod_reward, step = 0, 0, 0
        state = env.reset()
        while step < 600:
            step += 1
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            # modify reward to learn faster
            # modified_reward = reward
            # modified_reward = abs(state[0] + 0.5)
            # modified_reward = reward + (math.exp(abs(state[0] + 0.5)))
            modified_reward = reward + (abs(state[0]+0.5))/(env.max_position - env.min_position)
            mod_reward += modified_reward
            packed = Transition._make([state, action, modified_reward, next_state, done])
            td_error = agent.get_td_error(packed)
            agent.add_to_replay_buffer(td_error, packed)
            # update net if replay_buffer is full
            memory_cnt += 1
            if memory_cnt > 1024:
                loss.append(agent.update())
            if done:
                break
            state = next_state
        print(i, " episode reward ", total_reward, " modified reward", mod_reward)
        # test_rewards.append(test(agent, env))
        train_rewards.append(total_reward)
        mod_rewards.append(mod_reward)
    return loss, test_rewards, train_rewards, mod_rewards


def train_in_diff_setting(agent, env, episode_num, setting):
    loss_dict, train_reward_dict = {}, {}
    loss, test_rewards, train_rewards, mod_rewards = train(agent, env, episode_num)
    saved = {'loss': loss, 'test': test_rewards, 'train': train_rewards, 'mod': mod_rewards}
    # pickle.dump(saved, open('data/' + setting + '.txt', 'wb'))
    # agent.save('model/' + setting + str(episode_num) + '.pth')
    loss_dict[setting] = loss
    train_reward_dict[setting] = train_rewards
    plot_data(loss_dict, episode_num, 'Loss', 'img/plot' + setting)
    plot_data(train_reward_dict, episode_num, 'Train Reward', 'img/plot' + setting)


def main():
    episode_num = 250
    env = gym.envs.make("MountainCar-v0").unwrapped
    # # basic DQN
    agent = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                epsilon=0.1, memory_size=20000, batch_size=256, target_update=100)
                # ,use_dueling=True, use_double=True, use_per=True)
    train_in_diff_setting(agent, env, episode_num, 'DQN')
    # # PER DQN
    # agent = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
    #             use_per=True, hidden_dim=128, batch_size=128)
    # train_in_diff_setting(agent, env, episode_num, 'PER_DQN')
    # # Double DQN
    # agent = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
    #             use_double=True)
    # train_in_diff_setting(agent, env, episode_num, 'Double_DQN')
    # Dueling
    # agent = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
    #             use_dueling=True, hidden_dim=100, batch_size=128)
    # train_in_diff_setting(agent, env, episode_num, 'Dueling_DQN')
    # # double dueling PER DQN
    # agent = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
    #             use_double=True, use_dueling=True, use_per=True, hidden_dim=128)
    # train_in_diff_setting(agent, env, episode_num, 'Double_Dueling_PER_DQN')

    env.close()


if __name__ == '__main__':
    main()
