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

    def __init__(self, capacity, transition_size):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros((capacity, transition_size), dtype=np.float32, order='C')  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        for d in data:
            tree_idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer, :] = d  # update data_frame
            self.update(tree_idx, p)  # update tree_frame, add this p (maxp) for all transitions

            self.data_pointer += 1
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
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
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, data_idx

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.00001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, batch_size, transition_size):
        self.tree = SumTree(capacity, transition_size)
        self.batch_size = batch_size
        self.b_idx, self.b_memory, self.ISWeights = \
            np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, transition_size), dtype=np.float32), \
            np.empty((batch_size,), dtype=np.float32)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:  # add maxp for all transitions
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self):
        pri_seg = self.tree.total_p / self.batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        data_indices = np.empty_like(self.b_idx)
        for i in range(self.batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            self.b_idx[i], data_indices[i] = self.tree.get_leaf(v)

        np.take(self.tree.tree, self.b_idx, axis=0, out=self.ISWeights)
        np.power(self.ISWeights / self.tree.total_p / min_prob, -self.beta, out=self.ISWeights)
        np.take(self.tree.data, data_indices, axis=0, out=self.b_memory)
        return self.b_idx, self.b_memory, self.ISWeights[:, np.newaxis]

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = ps = abs_errors[:]
        np.minimum(abs_errors, self.abs_err_upper, out=clipped_errors)
        np.power(clipped_errors, self.alpha, out=ps)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


def init_w_b(layer):
    init.xavier_normal(layer.weight)
    init.constant(layer.bias, 0.1)


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound, name):
        super(Actor, self).__init__()
        self.name = name
        self.a_bound = a_bound
        layers = [256, 256]
        self.fc1 = nn.Linear(s_dim, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], a_dim)
        [init_w_b(l) for l in [self.fc1, self.fc2, self.out]]
        self.ac1 = nn.ReLU()
        self.tanh = nn.Tanh()
        self.scale = (self.a_bound[1] - self.a_bound[0]) / 2
        self.shift = self.a_bound[1] - self.scale
        if '_' in name:     # set target net to not bp
            self.eval()

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
        layers = [256, 256]
        self.w1s = Parameter(torch.Tensor(s_dim, layers[0]))
        self.w1a = Parameter(torch.Tensor(a_dim, layers[0]))
        [w.data.normal_(0.0, 0.01) for w in [self.w1a, self.w1s]]
        self.b1 = Parameter(torch.zeros(1, layers[0])+0.1)
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], 1)
        [init_w_b(l) for l in [self.fc2, self.out]]
        self.ac1 = nn.ReLU()
        if '_' in name:     # set target net to not bp
            self.eval()

    def forward(self, s, a):
        w1x = torch.mm(a, self.w1a) + torch.mm(s, self.w1s)
        net = self.ac1(w1x + self.b1.expand_as(w1x))
        net = self.ac1(self.fc2(net))
        s_value = self.out(net)
        return s_value


class DDPG(object):
    def __init__(self,
                 s_dim, a_dim, a_bound,
                 a_lr=0.001,c_lr=0.001,
                 tau=0.001, gamma=0.9,
                 memory_capacity=5000, batch_size=64,
                 train={'train': True, 'save_iter': None, 'load_point': -1},
                 model_dir='./torch_models',
                 ):

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.learn_counter = 0
        self.tau = tau
        self.gamma = gamma
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.train_ = train

        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.update_times = 10
        self.batch_holder = np.empty((self.batch_size*self.update_times, self.memory.shape[1]), dtype=np.float32)

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

    def learn(self, lock=None):
        # hard replacement
        self._soft_rep_target()
        self._check_save()

        indices = np.random.randint(self.memory_capacity, size=self.batch_size*self.update_times)
        if lock is not None: lock.acquire()
        np.take(self.memory, indices, axis=0, out=self.batch_holder)
        if lock is not None: lock.release()

        s, a, r, s_ = self.batch_holder[:, :self.s_dim], \
                      self.batch_holder[:, self.s_dim: self.s_dim + self.a_dim], \
                      self.batch_holder[:, -self.s_dim - 1: -self.s_dim], \
                      self.batch_holder[:, -self.s_dim:]
        Vs, Va, Vr, Vs_ = Variable(torch.from_numpy(s)), Variable(torch.from_numpy(a), requires_grad=True), \
                          Variable(torch.from_numpy(r)), Variable(torch.from_numpy(s_))
        for ut in range(self.update_times):
            self.learn_counter += 1
            Vbs, Vba, Vbr, Vbs_ = Vs[ut * self.batch_size: (ut + 1) * self.batch_size], \
                                  Va[ut * self.batch_size: (ut + 1) * self.batch_size], \
                                  Vr[ut * self.batch_size: (ut + 1) * self.batch_size], \
                                  Vs_[ut * self.batch_size: (ut + 1) * self.batch_size]

            target_q = Vbr + self.gamma * self.cnet_(Vbs_, self.anet_(Vbs_)).detach()    # not train
            c_loss = self.closs_func(self.cnet(Vbs, Vba), target_q)
            self.copt.zero_grad()
            c_loss.backward()
            self.copt.step()

            policy_loss = -self.cnet(Vbs, self.anet(Vbs)).mean()
            self.aopt.zero_grad()
            policy_loss.backward()
            self.aopt.step()

    def threadlearn(self, stop_event, lock=None):
        while not stop_event.is_set():
            self.learn(lock)

    def store_transition(self, s, a, r, s_):
        if a.ndim < 2:
            a = a[:, None]
        if r.ndim < 2:
            r = r[:, None]
        transition = np.concatenate((s, a, r, s_), 1)
        transition = np.ascontiguousarray(transition)
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
            if self.learn_counter % self.train_['save_iter'] == 0:
                self.save()
                print('\nSaved model for point=%i\n' % self.learn_counter)

    def save(self):
        for net in self.all_nets:
            torch.save(net.state_dict(), './torch_models/params%i_%s.pkl' % (self.learn_counter, net.name))  # save only the parameters

    def _soft_rep_target(self):
        for (aname, aparam), (cname, cparam) in zip(self.anet.state_dict().items(), self.cnet.state_dict().items()):
            self.anet_.state_dict()[aname].copy_((1-self.tau)*self.anet_.state_dict()[aname] + self.tau*aparam)
            self.cnet_.state_dict()[cname].copy_((1-self.tau)*self.cnet_.state_dict()[cname] + self.tau*cparam)


class DDPGPrioritizedReplay(DDPG):

    def __init__(self,
                 s_dim, a_dim, a_bound,
                 a_lr=0.001, c_lr=0.001,
                 tau=0.001, gamma=0.9,
                 memory_capacity=5000, batch_size=64,
                 train={'train': True, 'save_iter': None, 'load_point': -1},
                 model_dir='./model',
                 ):
        super(DDPGPrioritizedReplay, self).__init__(
            s_dim=s_dim, a_dim=a_dim, a_bound=a_bound,
            a_lr=a_lr, c_lr=c_lr,
            tau=tau, gamma=gamma,
            memory_capacity=memory_capacity, batch_size=batch_size,
            train=train, model_dir=model_dir,)
        transition_size = s_dim * 2 + a_dim + 1
        self.memory = Memory(capacity=memory_capacity, batch_size=batch_size, transition_size=transition_size)

    def learn(self, lock=None):
        # hard replacement
        self._soft_rep_target()
        self._check_save()

        for _ in range(10):
            self.learn_counter += 1
            if lock is not None: lock.acquire()
            tree_idx, bt, ISWeights = self.memory.sample()
            if lock is not None: lock.release()

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
            if lock is not None: lock.acquire()
            self.memory.batch_update(tree_idx, abs_errors)
            if lock is not None: lock.release()

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