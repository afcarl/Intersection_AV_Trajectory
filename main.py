from env import Env, CrashEnv
from tfDDPG import DDPG, DDPGPrioritizedReplay
# from torchDDPG import DDPG, DDPGPrioritizedReplay
import numpy as np
import time, sys, subprocess, os, signal, platform
import threading


TOTAL_LEARN_STEP = 200000
A_LR = 0.0001   # 0.0002
C_LR = 0.0001    # 0.0005
TAU = 0.002     # 0.005
GAMMA = 0.9    # must small
AVERAGE_H = None
SAFE_T_GAP = 0.8    # no use for now
RANDOM_LIGHT = True
LIGHT_P = 1500
MAX_CEP_STEP = 50
MEMORY_CAPACITY = 1000000  # should consist of several episodes
BATCH_SIZE = 64


TRAIN = {'train': 1, 'save_iter': None, 'load_point': -1, "threading": True}
MODEL_PARENT_DIR = './tf_models'
LOAD_PATH = './tf_models/1'

env = Env(light_p=LIGHT_P, ave_h=AVERAGE_H, random_light_dur=RANDOM_LIGHT, safe_t_gap=SAFE_T_GAP)
crash_env = CrashEnv(light_p=LIGHT_P, safe_t_gap=SAFE_T_GAP)
env.set_fps(1000)
MAX_EP_STEP = int(380 / env.dt)
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


def twork(RL, n_val, lock=None, stop_event=None):
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
            ma1 = env.car_info['v'][:env.ncs] <= 0
            ma2 = a.ravel() < 0
            a[(ma1 & ma2)] = 0.   # TODO: 0 acceleration for stop
            np.clip(np.random.normal(a, var, size=a.shape), *A_BOUND, out=a)     # clip according to bound
            s_, r, done, new_car_s = env.step(a.ravel())  # remove and add new cars

            ep_r += np.mean(r)

            if TRAIN["threading"]:
                lock.acquire()
                RL.store_transition(s, a, r, s_)
                lock.release()
            else:
                RL.store_transition(s, a, r, s_)
                RL.learn()

            var = max([0.99998 * var, .1])  # TODO: keep exploring
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
                RL.ep_r = ep_r/MAX_EP_STEP      # record for tensorboard
                print(
                    '%i' % n_val,
                    '| Ep: %i' % i_ep,
                    '| RunningR: %.0f' % running_r[-1] if len(running_r) > 0 else 0.,
                    '| Ep_r: %.0f' % ep_r,
                    '| Var: %.3f' % var,
                    '| T100: %.2f' % measure100,
                    '| LC: %i' % RL.sess.run(RL.global_step),
                    '%s' % ('| 1prio: %.2f' % (RL.memory.tree.total_p/RL.memory_capacity) if RL.__class__.__name__ != 'DDPG' else ''),
                )
                break
            s = new_car_s

    if TRAIN["threading"]:
        stop_event.set()

    save_path = MODEL_PARENT_DIR + '/%i' % n_val
    RL.save(path=save_path)
    np.save(save_path+'/running_r', np.array(running_r))


def crash_data(RL, stop_event, lock=None):
    # learning
    var = 7
    i_ep = 0
    while not stop_event.is_set():
        i_ep += 1
        s = crash_env.reset()
        ep_r = 0
        for step in range(MAX_CEP_STEP):
            # crash_env.render()
            a = RL.choose_action(s)
            ma1 = crash_env.car_info['v'][:crash_env.ncs] <= 0
            ma2 = a.ravel() < 0
            a[(ma1 & ma2)] = 0.   # TODO: 0 acceleration for stop
            a = np.clip(np.random.normal(a, var, size=a.shape), *A_BOUND).astype(np.float32)     # clip according to bound
            s_, r, done, new_car_s = crash_env.step(a.ravel())  # remove and add new cars

            ep_r += np.mean(r)
            if lock is not None: lock.acquire()
            RL.store_transition(s, a, r, s_)
            if lock is not None: lock.release()

            var = max([0.9999 * var, .5])  # TODO: keep exploring
            if done or step == MAX_CEP_STEP-1:
                break
            s = new_car_s


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
            if platform.system() == "Darwin":
                VALS = [1]
            else:
                VALS = [0]
        for i in VALS:
            RL = DDPG(
                s_dim=S_DIM, a_dim=A_DIM, a_bound=A_BOUND,
                a_lr=A_LR, c_lr=C_LR,
                tau=TAU, gamma=GAMMA,
                memory_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE,
                train=TRAIN, log_dir='log/%i' % i,
            )
            RL.reset()
            RL = fill_memory(RL)
            if len(sys.argv) > 2:
                if sys.argv[2] == 't':
                    pro = subprocess.Popen(["tensorboard", "--logdir", "log"])    # tensorboard
                    # import webbrowser
                    # webbrowser.open_new('http://localhost:6006/#scalars')

            if TRAIN["threading"]:
                stop_event = threading.Event()
                lock = threading.Lock()
                stop_event.clear()

                t1 = threading.Thread(target=RL.threadlearn, args=(stop_event,))
                # t2 = threading.Thread(target=crash_data, args=(RL, stop_event, lock))
                t1.start()
                # t2.start()
                twork(RL, i, lock, stop_event)
                t1.join()
                # t2.join()
            else:
                twork(RL, i)

            if len(sys.argv) > 2:
                if sys.argv[2] == 't':
                    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

    else:
        load()

