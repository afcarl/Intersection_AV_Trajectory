import tensorflow as tf
import numpy as np
import os
import shutil
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        for d in data:
            leaf_idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = d  # update data_frame
            self.update(leaf_idx, p)  # update tree_frame

            self.data_pointer += 1
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound - self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)   # set the max p for new p

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [None] * n, [None] * n, [None] * n
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights[i] = self.tree.capacity * prob
            batch_idx[i] = idx
            batch_memory[i] = data

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)


def init_w_b(layer):
    init.xavier_normal(layer.weight)
    init.constant(layer.bias, 0.01)


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound, name):
        super(Actor, self).__init__()
        self.name = name
        self.a_bound = a_bound
        layers = [100, 100]
        self.fc1 = nn.Linear(s_dim, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], a_dim)
        [init_w_b(l) for l in [self.fc1, self.fc2, self.out]]
        self.ac1 = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.scale = (self.a_bound[1] - self.a_bound[0]) / 2
        self.shift = self.a_bound[1] - self.scale

    def forward(self, s):
        net = self.ac1(self.fc1(s))
        net = self.ac1(self.fc2(net))
        net = self.tanh(self.out(net))
        actions = net * self.scale + self.shift
        return actions


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, name):
        super(Critic, self).__init__()
        self.name = name
        layers = [100, 100]
        self.w1s = Parameter(torch.Tensor(s_dim, layers[0]))
        self.w1a = Parameter(torch.Tensor(a_dim, layers[0]))
        [w.data.normal_(0.0, 0.01) for w in [self.w1a, self.w1s]]
        self.b1 = Parameter(torch.zeros(1, layers[0])+0.01)
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], 1)
        [init_w_b(l) for l in [self.fc2, self.out]]
        self.ac1 = nn.LeakyReLU()

    def forward(self, s, a):
        w1x = torch.mm(a, self.w1a) + torch.mm(s, self.w1s)
        net = self.ac1(w1x + self.b1.expand_as(w1x))
        net = self.ac1(self.fc2(net))
        s_value = self.out(net)
        return s_value


class DDPG(object):
    def __init__(self,
                 s_dim, a_dim, a_bound,
                 a_lr=0.001, a_replace_iter=600,
                 c_lr=0.001, c_replace_iter=500, gamma=0.9,
                 memory_capacity=5000, batch_size=64,
                 train={'train': True, 'save_iter': 10000, 'load_point': 400},
                 model_dir='./torch_models',
                 ):

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.a_replace_iter = a_replace_iter
        self.c_replace_iter = c_replace_iter
        self.replace_counter = 0
        self.gamma = gamma
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.train_ = train

        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.anet, self.anet_ = Actor(s_dim, a_dim, a_bound, 'anet'), Actor(s_dim, a_dim, a_bound, 'anet_')
        self.cnet, self.cnet_ = Critic(s_dim, a_dim, 'cnet'), Critic(s_dim, a_dim, 'cnet_')
        self.all_nets = [self.anet, self.anet_, self.cnet, self.cnet_]
        if not self.train_['train']:
            for net in self.all_nets:
                net.load_state_dict(torch.load('./torch_models/params%i_%s.pkl' % (self.train_['load_point'], net.name)))

        self.aopt = torch.optim.Adam(self.anet.parameters(), lr=a_lr)
        self.copt = torch.optim.Adam(self.cnet.parameters(), lr=c_lr)
        self.closs_func = torch.nn.MSELoss()

    def choose_action(self, s):
        s = Variable(torch.from_numpy(s).float())
        return self.anet(s).data.numpy()

    def learn(self):
        # hard replacement
        self._check_rep_target()
        self._check_save()

        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs, ba, br, bs_ = bt[:, :self.s_dim], bt[:, self.s_dim: self.s_dim + self.a_dim], \
                          bt[:, -self.s_dim - 1: -self.s_dim], bt[:, -self.s_dim:]
        Vbs, Vba, Vbr, Vbs_ = Variable(torch.from_numpy(bs)), Variable(torch.from_numpy(ba), requires_grad=True), \
                              Variable(torch.from_numpy(br)), Variable(torch.from_numpy(bs_))

        target_q = Vbr + self.gamma * self.cnet_(Vbs_, self.anet_(Vbs_)).detach()    # not train
        c_loss = self.closs_func(self.cnet(Vbs, Vba), target_q)
        self.copt.zero_grad()
        c_loss.backward()
        self.copt.step()

        policy_loss = -self.cnet(Vbs, self.anet(Vbs)).mean()
        self.aopt.zero_grad()
        policy_loss.backward()
        self.aopt.step()

    def store_transition(self, s, a, r, s_):
        if a.ndim < 2:
            a = a[:, np.newaxis]
        if r.ndim < 2:
            r = r[:, np.newaxis]
        transition = np.concatenate((s, a, r, s_), 1)
        p_ = self.pointer + transition.shape[0]
        if p_ <= self.memory_capacity:
            self.memory[self.pointer:p_, :] = transition
        else:
            p_ = p_ % self.memory_capacity
            self.memory[self.pointer:, :] = transition[:transition.shape[0] - p_, :]
            self.memory[:p_, :] = transition[-p_:, :]
        self.pointer = p_

    def _check_save(self):
        if self.train_['save_iter'] is not None:
            if self.replace_counter % self.train_['save_iter'] == 0:
                self.save()
                print('\nSaved model for point=%i\n' % self.replace_counter)

    def save(self):
        for net in self.all_nets:
            torch.save(net.state_dict(), './torch_models/params%i_%s.pkl' % (self.replace_counter, net.name))  # save only the parameters

    def _check_rep_target(self):
        if self.replace_counter % self.a_replace_iter == 0:
            self.anet_.load_state_dict(self.anet.state_dict())
        if self.replace_counter % self.c_replace_iter == 0:
            self.cnet_.load_state_dict(self.cnet.state_dict())
        self.replace_counter += 1


class DDPGPrioritizedReplay(DDPG):

    def __init__(self,
                 s_dim, a_dim, a_bound,
                 a_lr=0.001, a_replace_iter=600,
                 c_lr=0.001, c_replace_iter=500, gamma=0.9,
                 memory_capacity=5000, batch_size=64,
                 train={'train': True, 'save_iter': 10000, 'load_point': 400},
                 model_dir='./model',
                 ):
        super(DDPGPrioritizedReplay, self).__init__(
            s_dim=s_dim, a_dim=a_dim, a_bound=a_bound,
            a_lr=a_lr, a_replace_iter=a_replace_iter,
            c_lr=c_lr, c_replace_iter=c_replace_iter, gamma=gamma,
            memory_capacity=memory_capacity, batch_size=batch_size,
            train=train, model_dir=model_dir,)
        self.memory = Memory(capacity=memory_capacity)

    def learn(self):
        # hard replacement
        self._check_rep_target()
        self._check_save()

        tree_idx, bt, ISWeights = self.memory.sample(self.batch_size)

        bs, ba, br, bs_ = bt[:, :self.s_dim], bt[:, self.s_dim: self.s_dim + self.a_dim], \
                          bt[:, -self.s_dim - 1: -self.s_dim], bt[:, -self.s_dim:]
        Vbs, Vba, Vbr, Vbs_, VISW = \
            Variable(torch.from_numpy(bs).float()), Variable(torch.from_numpy(ba).float()), \
            Variable(torch.from_numpy(br).float()), Variable(torch.from_numpy(bs_).float()),\
            Variable(torch.from_numpy(ISWeights).float())

        target_q = Vbr + self.gamma * self.cnet_(Vbs_, self.anet_(Vbs_)).detach()  # not train
        td_errors = self.cnet(Vbs, Vba) - target_q

        # update priority
        abs_errors = torch.abs(td_errors).data.numpy()
        for i in range(len(tree_idx)):  # update priority
            idx = tree_idx[i]
            self.memory.update(idx, abs_errors[i])

        c_loss = torch.mean(VISW * torch.pow(td_errors, 2))
        self.copt.zero_grad()
        c_loss.backward()
        self.copt.step()

        policy_loss = -self.cnet(Vbs, self.anet(Vbs)).mean()
        self.aopt.zero_grad()
        policy_loss.backward()
        self.aopt.step()

    def store_transition(self, s, a, r, s_):
        if a.ndim < 2:
            a = a[:, np.newaxis]
        if r.ndim < 2:
            r = r[:, np.newaxis]
        transitions = np.concatenate((s, a, r, s_), 1)
        self.memory.store(transitions.astype(np.float32))


if __name__ == '__main__':
    import gym

    MAX_EPISODES = 70
    MAX_EP_STEPS = 400
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    GAMMA = 0.9  # reward discount
    TAU = 0.01  # Soft update for target param, but this is computationally expansive
    # so we use replace_iter instead
    REPLACE_ITER_A = 500
    REPLACE_ITER_C = 300
    MEMORY_CAPACITY = 7000
    BATCH_SIZE = 32

    RENDER = False
    OUTPUT_GRAPH = True
    ENV_NAME = 'Pendulum-v0'
    var = 3
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [-2, 2]
    ddpg = DDPG(state_dim, action_dim, action_bound, LR_A, REPLACE_ITER_A,
                LR_C, REPLACE_ITER_C, GAMMA, MEMORY_CAPACITY, BATCH_SIZE)
    counter = 0
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):

            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s[np.newaxis, :])[0]
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s[np.newaxis, :], a, np.array([r / 10]), s_[np.newaxis, :])

            if counter >= MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                ddpg.learn()

            counter += 1
            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -1000:
                    RENDER = True
                break