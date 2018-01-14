from env import Env
from tfDDPG import DDPG
import numpy as np
import subprocess, os, signal
from datetime import datetime
import threading as td
import multiprocessing as mp
import argparse


TOTAL_LEARN_STEP = 250000
A_LR = 0.0001   # 0.0002
C_LR = 0.0001    # 0.0005
TAU = 0.001     # 0.005
GAMMA = 0.9    # must small
AVERAGE_H = None
SAFE_T_GAP = 0.8    # no use for now
RANDOM_LIGHT = True
LIGHT_P = 1500
MAX_CEP_STEP = 50
MEMORY_CAPACITY = 500000  # should consist of several episodes
BATCH_SIZE = 64
N_WORKER = 6


TRAIN = {'train': 1, 'save_iter': None, 'load_point': -1, "threading": True}
MODEL_PARENT_DIR = './tf_models'
LOAD_PATH = './tf_models/1'

env = Env(light_p=LIGHT_P, ave_h=AVERAGE_H, random_light_dur=RANDOM_LIGHT, safe_t_gap=SAFE_T_GAP)
env.set_fps(1000)
MAX_EP_STEP = int(380 / env.dt)
A_DIM = env.action_dim
S_DIM = env.state_dim
A_BOUND = env.action_bound
del env


def p_fill_memory(m_counter, lock):
    env = Env(light_p=LIGHT_P, ave_h=AVERAGE_H, random_light_dur=RANDOM_LIGHT, safe_t_gap=SAFE_T_GAP)
    env.set_fps(1000)
    trans = []
    while True:
        s = env.reset()
        while True:
            a = env.sample_action()
            s_, r, done, new_car_s = env.step(a.ravel())  # remove and add new cars
            trans.append((s, a, r, s_))
            with lock:
                m_counter.value += len(s)
            s = new_car_s
            if done:
                break
        if m_counter.value >= MEMORY_CAPACITY:
            break
    return trans


def p_rollout(rl, lock):
    global i_ep, roll_step, r_his, var
    env = Env(light_p=LIGHT_P, ave_h=AVERAGE_H, random_light_dur=RANDOM_LIGHT, safe_t_gap=SAFE_T_GAP)
    while not IS_DONE:
        s = env.reset()
        ep_r = 0
        for step in range(MAX_EP_STEP):
            a = rl.choose_action(s)
            ma1 = env.car_info['v'][:env.ncs] <= 0
            ma2 = a.ravel() < 0
            a[(ma1 & ma2)] = 0.  # TODO: 0 acceleration for stop
            np.clip(np.random.normal(a, var, size=a.shape), *A_BOUND, out=a)  # clip according to bound
            s_, r, done, new_car_s = env.step(a.ravel())  # remove and add new cars

            ep_r += np.mean(r)

            with lock:
                roll_step += 1
                rl.store_transition(s, a, r, s_)

            var = max([0.999 * var, .1])  # TODO: keep exploring

            if done or step == MAX_EP_STEP - 1:
                r_his.append(ep_r)
                rl.ep_r = ep_r / MAX_EP_STEP  # record for tensorboard
                with lock:
                    i_ep += 1
                msg = ''.join([
                    '%i' % model_n,
                    '| Ep: %i' % i_ep,
                    '| Ep_r: %.0f' % ep_r,
                    '| Var: %.3f' % var,
                    '| LC: %i' % rl.sess.run(rl.global_step),
                    '| roll_step: %i' % roll_step,
                ])
                print_time_msg(msg)
                break
            s = new_car_s


def print_time_msg(msg):
    time_cost = datetime.now().replace(microsecond=0) - BASE_TIME.replace(microsecond=0)
    print(time_cost, msg)


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


def main(model_n, tb):
    if tb:
        pro = subprocess.Popen(["tensorboard", "--logdir", "log"])  # tensorboard

    global BASE_TIME, IS_DONE, i_ep, roll_step, r_his, var
    BASE_TIME = datetime.now()
    print_time_msg('Start', )
    rl = DDPG(s_dim=S_DIM, a_dim=A_DIM, a_bound=A_BOUND, a_lr=A_LR, c_lr=C_LR,
              tau=TAU, gamma=GAMMA, memory_capacity=MEMORY_CAPACITY,
              batch_size=BATCH_SIZE, train=TRAIN, log_dir='log/%i' % model_n, )
    rl.reset()

    manager = mp.Manager()
    m_counter = manager.Value('i', 0)
    lock = manager.Lock()
    print_time_msg('Filling memory...')
    pool = mp.Pool()
    filling_jobs = [pool.apply_async(func=p_fill_memory, args=(m_counter, lock)) for _ in range(N_WORKER)]
    all_trans = [j.get() for j in filling_jobs]
    for trans in all_trans:
        [rl.store_transition(s, a, r, s_) for s, a, r, s_ in trans]

    print_time_msg('Parallel training...')
    lock = td.Lock()
    var = 7
    roll_step = 0
    i_ep = 0
    r_his = []
    IS_DONE = False

    # rollout in env
    ts_train = [td.Thread(target=p_rollout, args=(rl, lock)) for _ in range(1)]
    [t.start() for t in ts_train]

    # training rl
    while rl.sess.run(rl.global_step) < TOTAL_LEARN_STEP:
        with lock:
            rl.learn()
    IS_DONE = True

    [t.join() for t in ts_train]
    print_time_msg('done')

    rl.save('./tf_models/%i' % model_n)
    if tb:  # kill tensorboard
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_n", type=int, nargs='+', choices=range(6), default=[6],
                        help="store model n", )
    parser.add_argument('-t', '--tensorboard', action='store_true',
                        help="to use tensorboard or not")
    args = parser.parse_args()
    for model_n in args.model_n:
        main(model_n, args.tensorboard)







