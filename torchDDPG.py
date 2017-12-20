import numpy as np
import os
import shutil
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init
from memory import Memory


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
        self.memory = np.empty(memory_capacity,
                               dtype=[('s', np.float32, (s_dim,)), ('a', np.float32, (a_dim,)),
                                      ('r', np.float32, (1,)), ('s_', np.float32, (s_dim,))])
        self.pointer = 0
        self.update_times = 10
        self.batch_holder = np.empty(self.batch_size*self.update_times,
                               dtype=[('s', np.float32, (s_dim,)), ('a', np.float32, (a_dim,)),
                                      ('r', np.float32, (1,)), ('s_', np.float32, (s_dim,))])

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

        s, a, r, s_ = self.batch_holder['s'], self.batch_holder['a'], self.batch_holder['r'], self.batch_holder['s_']
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
        for item in [s,a,r,s_]:
            assert item.dtype == np.float32
        if a.ndim < 2:
            a = a[:, None]
        if r.ndim < 2:
            r = r[:, None]
        s = np.ascontiguousarray(s)
        s_ = np.ascontiguousarray(s_)
        p_ = self.pointer + r.size
        if p_ <= self.memory_capacity:
            self.memory['s'][self.pointer:p_] = s
            self.memory['a'][self.pointer:p_] = a
            self.memory['r'][self.pointer:p_] = r
            self.memory['s_'][self.pointer:p_] = s_
        else:
            p_ = p_ % self.memory_capacity
            self.memory['s'][self.pointer:] = s[:r.size - p_]
            self.memory['a'][self.pointer:] = a[:r.size - p_]
            self.memory['r'][self.pointer:] = r[:r.size - p_]
            self.memory['s_'][self.pointer:] = s_[:r.size - p_]
            self.memory['s'][:p_] = s[-p_:]
            self.memory['a'][:p_] = a[-p_:]
            self.memory['r'][:p_] = r[-p_:]
            self.memory['s_'][:p_] = s_[-p_:]
        self.pointer = p_

    def _check_save(self):
        if self.train_['save_iter'] is not None:
            if self.learn_counter % self.train_['save_iter'] == 0:
                self.save()
                print('\nSaved model for point=%i\n' % self.learn_counter)

    def save(self):
        if os.path.isdir(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.mkdir(self.model_dir)
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
        self.memory = Memory(capacity=memory_capacity, batch_size=batch_size, s_dim=s_dim, a_dim=a_dim)

    def learn(self, lock=None):
        # hard replacement
        self._soft_rep_target()
        self._check_save()

        for _ in range(self.update_times):
            self.learn_counter += 1
            if lock is not None: lock.acquire()
            tree_idx, bt, ISWeights = self.memory.sample()

            bs, ba, br, bs_ = bt['s'], bt['a'], bt['r'], bt['s_']
            Vbs, Vba, Vbr, Vbs_, VISW = \
                Variable(torch.from_numpy(bs).float()), Variable(torch.from_numpy(ba).float()), \
                Variable(torch.from_numpy(br).float()), Variable(torch.from_numpy(bs_).float()),\
                Variable(torch.from_numpy(ISWeights).float())

            target_q = Vbr + self.gamma * self.cnet_(Vbs_, self.anet_(Vbs_)).detach()  # not train
            td_errors = self.cnet(Vbs, Vba) - target_q

            # update priority
            abs_errors = torch.abs(td_errors).data.numpy()
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
            a = a[:, None]
        if r.ndim < 2:
            r = r[:, None]
        s = np.ascontiguousarray(s)
        s_ = np.ascontiguousarray(s_)
        self.memory.store(s, a, r, s_)


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