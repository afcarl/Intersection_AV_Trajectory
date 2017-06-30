from env import Env
# from DDPG import DDPG, DDPGPrioritizedReplay
from torchDDPG import DDPG, DDPGPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt
import sys


TOTAL_EP = 300
A_LR = 0.0001
C_LR = 0.0001
A_REPLACE_ITER = 1000
C_REPLACE_ITER = 1000
GAMMA = 0.9
MEMORY_CAPACITY = 20000  # should consist of several episodes
BATCH_SIZE = 64
MAX_EP_STEP = 5000
TRAIN = {'train': True, 'save_iter': 50000, 'load_point': 250000}

env = Env()
env.set_fps(1000)
A_DIM = env.action_dim
S_DIM = env.state_dim
A_BOUND = env.action_bound

RL = DDPG(
        s_dim=S_DIM, a_dim=A_DIM, a_bound=A_BOUND,
        a_lr=A_LR, a_replace_iter=A_REPLACE_ITER,
        c_lr=C_LR, c_replace_iter=C_REPLACE_ITER, gamma=GAMMA,
        memory_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE,
        train=TRAIN,
    )


def train():
    print('Storing transitions....')
    m_counter = 0
    while True:
        s = env.reset()
        while True:
            a = env.sample_action()
            s_, r, done = env.step(a.flatten())
            RL.store_transition(s, a, r, s_)
            m_counter += len(s)
            s = env.update_s_(s_)
            if done:
                break
        if m_counter >= RL.memory_capacity:
            break

    # learning
    global_step = 0
    var = 3
    running_r = []
    print('\nStart Training')
    for i_ep in range(TOTAL_EP):
        s = env.reset()
        ep_r = 0
        for step in range(MAX_EP_STEP):
            if i_ep >= 70:
                env.render()
            a = RL.choose_action(s)
            a = np.clip(np.random.normal(a, var, size=a.shape), *A_BOUND)     # clip according to bound
            s_, r, done = env.step(a.flatten())

            ep_r += np.mean(r)
            RL.store_transition(s, a, r, s_)

            var = max([0.9999 * var, 0.1])  # keep exploring
            RL.learn()
            global_step += 1
            if done or step == MAX_EP_STEP-1:
                if len(running_r) == 0:
                    running_r.append(ep_r)
                else:
                    running_r.append(0.99*running_r[-1]+0.01*ep_r)
                print(
                    'Ep: %i' % i_ep,
                    '| RunningR: %.2f' % running_r[-1] if len(running_r) > 0 else 0.,
                    '| Ep_r: %.2f' % ep_r,
                    '| Var: %.3f' % var,
                )
                break
            s = env.update_s_(s_)

    RL.save()
    # np.save(model_dir+'/running_r', np.array(running_r))
    # plot_running_r(5)


def load():
    env.set_fps(20)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = RL.choose_action(s)
            s_, r, done = env.step(a.flatten())
            s = env.update_s_(s_)
            if done:
                break


def plot_running_r(n_car):
    running_r = np.load('./model/'+str(n_car)+'_car/running_r.npy')
    plt.plot(np.array(running_r))
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TRAIN['train'] = int(sys.argv[1])
    print('Training..' if TRAIN['train'] else 'Testing..')
    if TRAIN['train']:
        train()
    else:
        load()

