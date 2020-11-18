import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import os, math, gym
import pickle
import Plotting
from SharedOptimizer import SharedRMSprop
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# limit the number of threads, for pytorch will use as many threads as it can


class ACNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(ACNet, self).__init__()
        self.state_dim, self.action_dim = state_space, action_space
        # Actor net
        self.p1 = nn.Linear(state_space, 200)
        self.mu = nn.Linear(200, action_space)     # parameter mu of normal distribution
        self.sigma = nn.Linear(200, action_space)  # parameter sigma of normal distribution
        self.distribution = torch.distributions.Normal
        # Critic net
        self.v1 = nn.Linear(state_space, 100)
        self.value = nn.Linear(100, 1)
        # initialization
        for layer in [self.p1, self.mu, self.sigma, self.v1, self.value]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        p1 = nn.functional.relu6(self.p1(x))
        mu = 2 * torch.tanh(self.mu(p1))
        sigma = nn.functional.softplus(self.sigma(p1)) + 0.001  # avoid 0
        v1 = nn.functional.relu6(self.v1(x))
        values = self.value(v1)
        return mu, sigma, values

    def choose_action(self, state):
        self.training = False
        mu, sigma, value = self.forward(state)  # shape of mu/sigma: (1, action_dim)
        action = self.distribution(mu.view(1, ).data, sigma.view(1, ).data).sample()
        return action.numpy()

    def calculate_loss(self, states, actions, v_t):
        self.train()
        mu, sigma, values = self.forward(states)
        td_error = v_t - values
        a_loss = td_error.pow(2)    # loss for Actor net

        m = self.distribution(mu, sigma)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = m.log_prob(actions) * td_error.detach() + 0.005 * entropy
        c_loss = -exp_v             # loss for Critic net
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.95, 0.999), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas,
                                         eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Worker(mp.Process):
    def __init__(self, env_n, global_net, opt, global_ep, global_ep_r, res_queue,
                 idx, max_ep=3000, max_ep_step=300, g_update_iter=5, gamma=0.9):
        super(Worker, self).__init__()
        self.env = gym.make(env_n).unwrapped
        self.l_net = ACNet(self.env.observation_space.shape[0], self.env.action_space.shape[0])  # local network
        self.g_net, self.opt = global_net, opt
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.name, self.gamma = 'worker%i' % idx, gamma
        self.max_ep, self.max_ep_step, self.g_update_iter = max_ep, max_ep_step, g_update_iter
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

    def run(self):
        while self.g_ep.value < self.max_ep:
            s = self.env.reset()
            ep_r, done, t = 0.0, False, 1
            while not done:
                a = self.l_net.choose_action(torch.tensor(s[None, :]).float())
                s_, r, done, _ = self.env.step(a.clip(-2, 2))  # action range:[-2, 2]
                if t == self.max_ep_step:   done = True
                ep_r += r
                self.buffer_a.append(a)
                self.buffer_s.append(s)
                self.buffer_r.append((r + 8.1) / 8.1)    # normalize
                # self.buffer_r.append(r)
                if t % self.g_update_iter == 0 or done:  # update global and assign to local net
                    self.push_and_pull(done, s_)         # sync with global net
                    self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
                s = s_
                t += 1
            self.record(ep_r)
        self.res_queue.put(None)

    def push_and_pull(self, done, s_):
        v_s_ = 0 if done else self.l_net.forward(torch.tensor(s_[None, :]).float())[-1].data.numpy()[0, 0]
        buffer_v_target = []
        for r in self.buffer_r[::-1]:  # reverse buffer r
            v_s_ = r + self.gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        loss = self.l_net.calculate_loss(torch.tensor(np.vstack(self.buffer_s)).float(),
            torch.tensor(np.vstack(self.buffer_a)).float(),
            torch.tensor(np.array(buffer_v_target)[:, None]).float())
        # calculate local gradients and push local parameters to global
        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.l_net.parameters(), self.g_net.parameters()):
            gp._grad = lp.grad
        self.opt.step()
        self.l_net.load_state_dict(self.g_net.state_dict())    # copy global parameters

    def record(self, ep_r):
        with self.g_ep.get_lock():
            self.g_ep.value += 1
        with self.g_ep_r.get_lock():
            if self.g_ep_r.value == 0.:
                self.g_ep_r.value = ep_r
            else:
                self.g_ep_r.value = self.g_ep_r.value * 0.9 + ep_r * 0.1
        self.res_queue.put(self.g_ep_r.value)
        print(self.name, "Ep:", self.g_ep.value, "| Ep_r: %.0f" % self.g_ep_r.value)


def train(env_name):
    rep_time = 5
    env = gym.make(env_name)
    state_space, action_space = env.observation_space.shape[0], env.action_space.shape[0]
    total_res = []
    for i in range(rep_time):
        global_net = ACNet(state_space, action_space)
        global_net.share_memory()
        opt = SharedAdam(global_net.parameters())
        # opt = torch.optim.Adam(global_net.parameters(), lr=1e-4, betas=(0.95, 0.999))
        # opt = torch.optim.RMSprop(global_net.parameters(), lr=1e-4, alpha=0.99)
        # opt = SharedRMSprop(global_net.parameters(), lr=1e-4, alpha=0.999)
        global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
        workers = [Worker(env_name, global_net, opt, global_ep, global_ep_r, res_queue, idx=i,
                          g_update_iter=5) for i in range(mp.cpu_count())]
        [w.start() for w in workers]
        res = []  # record episode reward to plot
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]
        total_res.append(res)

    file_name = 'SharedAdam_3000ep_300sp_5gst_8worker.txt'
    pickle.dump(total_res, open(file_name, 'wb'))
    Plotting.plot_one_ac(file_name)


if __name__ == '__main__':
    env_name = 'Pendulum-v0'
    train(env_name)

