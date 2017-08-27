from env import Env
from tfDDPG import DDPG, DDPGPrioritizedReplay
# from torchDDPG import DDPG, DDPGPrioritizedReplay
import numpy as np
import time, threading, sys, subprocess, os, signal


# TOTAL_EP = 900 # 800
TOTAL_LEARN_STEP = 250000
A_LR = 0.0003   # 0.0005
C_LR = 0.0005    # 0.001
TAU = 0.005     # 0.005
GAMMA = 0.95    # 0.95
MEMORY_CAPACITY = 200000  # should consist of several episodes
BATCH_SIZE = 32
MAX_EP_STEP = 1000
RANDOM_LIGHT = True
MAX_P = 1200
TRAIN = {'train': True, 'save_iter': None, 'load_point': -1}
MODEL_PARENT_DIR = './tf_models'
LOAD_PATH = './tf_models/0'

env = Env(max_p=MAX_P, random_light_dur=RANDOM_LIGHT)
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


def twork(RL, stop_event, n_val, lock=None):
    # learning
    global_step = 0
    measure100 = 0.
    t0 = time.time()
    var = 7
    running_r = []
    print('\nStart Training')
    i_ep = 0
    while RL.learn_counter < TOTAL_LEARN_STEP:
        i_ep += 1
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
                    running_r.append(0.95*running_r[-1]+0.05*ep_r)
                RL.ep_r = ep_r      # record for tensorboard
                print(
                    '%i' % n_val,
                    '| Ep: %i' % i_ep,
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
    save_path = MODEL_PARENT_DIR + '/%i' % n_val
    RL.save(path=save_path)
    np.save(save_path+'/running_r', np.array(running_r))


def load():
    RL = DDPG(
        s_dim=S_DIM, a_dim=A_DIM, a_bound=A_BOUND,
        train=TRAIN, model_dir=LOAD_PATH, output_graph=False,
    )
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

if __name__ == '__main__':
    print('Training..' if TRAIN['train'] else 'Testing..')

    if TRAIN['train']:
        if len(sys.argv) > 1:
            VALS = [int(i) for i in list(sys.argv[1])]
        else:
            VALS = [6]
        for i in VALS:
            RL = DDPG(
                s_dim=S_DIM, a_dim=A_DIM, a_bound=A_BOUND,
                a_lr=A_LR, c_lr=C_LR,
                tau=TAU, gamma=GAMMA,
                memory_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE,
                train=TRAIN, log_dir='log/%i' % i,
            )
            RL = fill_memory(RL)
            if len(sys.argv) > 2:
                if sys.argv[2] == 't':
                    pro = subprocess.Popen(["tensorboard", "--logdir", "log"])    # tensorboard
            stop_event = threading.Event()
            lock = threading.Lock()
            stop_event.clear()
            t = threading.Thread(target=RL.threadlearn, args=(stop_event, lock))
            t.start()
            twork(RL, stop_event, i, lock)
            t.join()
            if len(sys.argv) > 2:
                if sys.argv[2] == 't':
                    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

    else:
        load()

