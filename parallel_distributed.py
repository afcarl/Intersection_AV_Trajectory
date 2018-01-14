from env import Env
from tfDDPG import DDPG
import numpy as np
import subprocess, os, signal, argparse, shutil
from datetime import datetime
import multiprocessing as mp
import tensorflow as tf


TOTAL_LEARN_STEP = 250000
A_LR = 0.0001   # 0.0002
C_LR = 0.0001    # 0.0005
TAU = 0.002     # 0.005
GAMMA = 0.9    # must small
AVERAGE_H = None
SAFE_T_GAP = 0.8    # no use for now
RANDOM_LIGHT = True
LIGHT_P = 1500
MAX_CEP_STEP = 50
MEMORY_CAPACITY = 100000  # should consist of several episodes
BATCH_SIZE = 64
BASE_TIME = datetime.now().replace(microsecond=0)
N_SERVER = 1
N_WORKER = 2
CLUSTER_DICT = {'ps': ['localhost:22%s' % str(22-i).zfill(2) for i in range(N_SERVER)],
                'worker': ['localhost:22%s' % str(23+i).zfill(2) for i in range(N_WORKER)]}


TRAIN = {'train': 1, 'save_iter': None, 'load_point': -1}
MODEL_PARENT_DIR = './tf_models'
LOAD_PATH = './tf_models/1'

env = Env(light_p=LIGHT_P, ave_h=AVERAGE_H, random_light_dur=RANDOM_LIGHT, safe_t_gap=SAFE_T_GAP)
MAX_EP_STEP = int(380 / env.dt)
A_DIM = env.action_dim
S_DIM = env.state_dim
A_BOUND = env.action_bound
del env


def connect_server(job_name, task_index):
    cluster = tf.train.ClusterSpec(CLUSTER_DICT)
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    return cluster, server


def server_on(task_index):
    _, server = connect_server('ps', task_index)
    print('Server %i ready' % task_index)
    server.join()


def rollout(task_index, model_n, lock, queue, i_ep):
    if task_index != 0:
        env = Env(light_p=LIGHT_P, ave_h=AVERAGE_H, random_light_dur=RANDOM_LIGHT, safe_t_gap=SAFE_T_GAP)
        env.set_fps(1000)
    cluster, server = connect_server('worker', task_index)
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % task_index,
            cluster=cluster)):
        rl = DDPG(s_dim=S_DIM, a_dim=A_DIM, a_bound=A_BOUND, a_lr=A_LR, c_lr=C_LR,
                  tau=TAU, gamma=GAMMA, memory_capacity=MEMORY_CAPACITY,
                  batch_size=BATCH_SIZE, train=TRAIN, log_dir='log/%i' % model_n, )
    hooks = [tf.train.StopAtStepHook(last_step=TOTAL_LEARN_STEP)]
    with tf.train.MonitoredTrainingSession(
            master=server.target, is_chief=True,
            # checkpoint_dir='./tmp',
            # save_summaries_steps=200,
            hooks=hooks, ) as sess:
        print('Start ', task_index)
        rl.sess = sess
        var = 6
        count = 0
        while not sess.should_stop():
            if task_index == 0:
                try:
                    s, a, r, s_ = queue.get(block=False)
                    rl.store_transition(s, a, r, s_)
                    count += len(s)
                except:
                    pass
                if count > rl.memory_capacity:
                    rl.learn()
            else:
                s = env.reset()
                ep_r = 0
                for step in range(MAX_EP_STEP):
                    # if task_index == 1:
                    #     env.render()
                    a = rl.choose_action(s)
                    ma1 = env.car_info['v'][:env.ncs] <= 0
                    ma2 = a.ravel() < 0
                    a[(ma1 & ma2)] = 0.  # TODO: 0 acceleration for stop
                    np.clip(np.random.normal(a, var, size=a.shape), *A_BOUND, out=a)  # clip according to bound
                    s_, r, done, new_car_s = env.step(a.ravel())  # remove and add new cars

                    ep_r += np.mean(r)

                    queue.put((s, a, r, s_))

                    var = max([0.999 * var, .1])  # TODO: keep exploring

                    if done or step == MAX_EP_STEP - 1:
                        rl.ep_r = ep_r / MAX_EP_STEP  # record for tensorboard
                        with lock:
                            i_ep.value += 1
                        msg = ''.join([
                            '%i' % model_n,
                            '| Task: %i' % task_index,
                            '| Ep: %i' % i_ep.value,
                            '| Ep_r: %.0f' % ep_r,
                            '| Var: %.3f' % var,
                            '| LC: %i' % sess.run(rl.global_step),
                        ])
                        print_time_msg(msg)
                        break
                    s = new_car_s



def print_time_msg(msg):
    time_cost = datetime.now().replace(microsecond=0) - BASE_TIME
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

    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    print_time_msg('Launch servers...')
    ps = [mp.Process(target=server_on, args=(i,)) for i in range(1)]
    [p.start() for p in ps]

    lock = mp.Lock()
    queue = mp.Queue(maxsize=100)
    i_ep = mp.Value('i', 0)

    # rollout in env
    print_time_msg('Lunch workers')
    workers = [mp.Process(target=rollout, args=(i, model_n, lock, queue, i_ep)) for i in range(N_WORKER)]

    [w.start() for w in workers]
    [w.join() for w in workers]
    print_time_msg('done')

    if tb:
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







