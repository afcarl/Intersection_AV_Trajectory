from env import Env
from tfDDPG import DDPG, DDPGPrioritizedReplay
# from torchDDPG import DDPG, DDPGPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt
import time, threading


TOTAL_EP = 600
A_LR = 0.0005
C_LR = 0.001
TAU = 0.005
GAMMA = 0.95
MEMORY_CAPACITY = 100000  # should consist of several episodes
BATCH_SIZE = 32
MAX_EP_STEP = 1500
TRAIN = {'train': False, 'save_iter': None, 'load_point': -1}

env = Env()
env.set_fps(1000)
A_DIM = env.action_dim
S_DIM = env.state_dim
A_BOUND = env.action_bound


def fill_memory(RL):
    print('Filling transitions....')
    m_counter = 0
    while True:
        s = env.reset()
        while True:
            a = env.sample_action()
            s_, r, done, new_car_s = env.step(a.ravel())    # remove and add new cars
            RL.store_transition(s, a, r, s_)
            m_counter += len(s)
            s = new_car_s
            if done:
                break
        if m_counter >= RL.memory_capacity:
            break
    return RL


def twork(RL, stop_event, lock=None,):
    # learning
    global_step = 0
    measure100 = 0.
    t0 = time.time()
    var = 7
    running_r = []
    print('\nStart Training')
    for i_ep in range(TOTAL_EP):
        s = env.reset()
        ep_r = 0
        for step in range(MAX_EP_STEP):
            env.render()
            a = RL.choose_action(s)
            a = np.clip(np.random.normal(a, var, size=a.shape), *A_BOUND).astype(np.float32)     # clip according to bound
            s_, r, done, new_car_s = env.step(a.ravel())  # remove and add new cars

            ep_r += np.mean(r)
            if lock is not None: lock.acquire()
            RL.store_transition(s, a, r, s_)
            if lock is not None: lock.release()

            var = max([0.99999 * var, 0.5])  # keep exploring
            global_step += 1
            if global_step % 100 == 0:      # record time
                t100 = time.time()
                measure100 = t100 - t0
                t0 = t100
            if done or step == MAX_EP_STEP-1:
                if len(running_r) == 0:
                    running_r.append(ep_r)
                else:
                    running_r.append(0.99*running_r[-1]+0.01*ep_r)
                RL.ep_r = ep_r
                print(
                    'Ep: %i' % i_ep,
                    '| RunningR: %.0f' % running_r[-1] if len(running_r) > 0 else 0.,
                    '| Ep_r: %.0f' % ep_r,
                    '| Var: %.3f' % var,
                    '| T100: %.2f' % measure100,
                    '| LC: %i' % RL.learn_counter,
                    '%s' % ('| 1prio: %.2f' % (RL.memory.tree.total_p/RL.memory_capacity) if RL.__class__.__name__ != 'DDPG' else ''),
                )
                break
            s = new_car_s
    stop_event.set()
    RL.save()
    # np.save(model_dir+'/running_r', np.array(running_r))
    # plot_running_r(5)


def load():
    env.set_fps(20)
    while True:
        s = env.reset()
        for t in range(MAX_EP_STEP):
            env.render()
            a = RL.choose_action(s)
            s_, r, done, new_s = env.step(a.ravel())
            s = new_s
            if done:
                break


def plot_running_r(n_car):
    running_r = np.load('./model/'+str(n_car)+'_car/running_r.npy')
    plt.plot(np.array(running_r))
    plt.show()


if __name__ == '__main__':
    print('Training..' if TRAIN['train'] else 'Testing..')
    RL = DDPG(
        s_dim=S_DIM, a_dim=A_DIM, a_bound=A_BOUND,
        a_lr=A_LR, c_lr=C_LR,
        tau=TAU, gamma=GAMMA,
        memory_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE,
        train=TRAIN,
    )
    if TRAIN['train']:
        RL = fill_memory(RL)
        stop_event = threading.Event()
        lock = threading.Lock()
        stop_event.clear()
        t = threading.Thread(target=RL.threadlearn, args=(stop_event, lock))
        t.start()
        twork(RL, stop_event, lock,)
        t.join()

    else:
        load()

