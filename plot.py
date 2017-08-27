from env import Env
from tfDDPG import DDPG, DDPGPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize
import sys, os

@vectorize('float32(float32, float32, float32)')
def IDM(vn, delta_vn, delta_xn):
    v0 = 120/3.6
    T_desired = 1.6
    a_max = 0.73
    b_max = 1.67
    s0 = 2.
    value = vn*T_desired+vn*delta_vn/(2*(a_max*b_max)**2)
    s_star = s0 + np.maximum(value, 0.)
    a = a_max*(1.-(vn/v0)**4-(s_star/delta_xn)**2)
    return a


def av_loop(RL, env, max_ep_step, return_light=True):
    t = 0
    pos = np.zeros((300, max_ep_step))
    pos[:] = pos.fill(np.nan)
    vel = pos.copy()
    if return_light:
        red_t, yellow_t = [], []
    while t < max_ep_step:
        s = env.reset()
        while True:
            # env.render()
            a = RL.choose_action(s)
            s_, r, done, new_s = env.step(a.ravel())
            s = new_s
            envp, envid, envv = env.car_info['p'][:env.ncs], env.car_info['id'][:env.ncs], env.car_info['v'][:env.ncs]
            pos[envid, t] = envp
            vel[envid, t] = envv

            if return_light:
                if env.is_red_light and (env.t_light >= env.light_duration['yellow']):
                    red_t.append(env.light_p)
                else:
                    red_t.append(np.nan)
                if env.is_red_light and (env.t_light < env.light_duration['yellow']):
                    yellow_t.append(env.light_p)
                else:
                    yellow_t.append(np.nan)

            t += 1
            if done or t == max_ep_step:
                break

    if return_light:
        return pos, vel, red_t, yellow_t
    else:
        return pos, vel


def mv_loop(env, max_ep_step):
    pos = np.zeros((300, max_ep_step))
    pos[:] = pos.fill(np.nan)
    vel = pos.copy()
    t = 0
    while t < max_ep_step:
        s = env.reset()
        while True:
            # env.render()
            # dx_norm( / 100m), dv_norm (/10), v_norm (/self.max_v), d2l (/1000)
            dx, dv, v, d2l = s[:, -3] * 100., s[:, -2] * 10., s[:, -1] * env.max_v, s[:, 2] * 1000
            if env.is_red_light:
                dv = np.where(dx > d2l, v, dv)
                np.minimum(dx, d2l, out=dx)
            a = IDM(v, dv, dx)
            s_, r, done, new_s = env.step(a)
            s = new_s
            envp, envid, envv = env.car_info['p'][:env.ncs], env.car_info['id'][:env.ncs], env.car_info['v'][:env.ncs]
            pos[envid, t] = envp
            vel[envid, t] = envv
            t += 1
            if done or t == max_ep_step:
                break
    return pos, vel


def plot_av_mv(max_ep_step):
    env = Env(max_p=MAX_P, ave_h=2, fix_start=True)
    env.set_fps(1000)
    rl = DDPG(
        s_dim=env.state_dim, a_dim=env.action_dim, a_bound=env.action_bound,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    # av test
    av_pos, av_vel, red_t, yellow_t = av_loop(rl, env, max_ep_step, return_light=True)
    # mv test
    mv_pos, mv_vel = mv_loop(env, max_ep_step)

    plt.figure(1, figsize=(6, 8))
    ax1 = plt.subplot(211)
    plt.title('(a) MVs benchmark')
    plt.plot(mv_pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')

    plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.title('(b) AVs case')
    plt.plot(av_pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, max_ep_step, 500), (np.arange(0, max_ep_step, 500)/10).astype(np.int32))
    plt.xlim(xmin=0, xmax=max_ep_step)
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.show()


def plot_av_diff_h(h_list, max_ep_step):
    env = Env(max_p=MAX_P, ave_h=2, fix_start=True)
    rl = DDPG(
        s_dim=env.state_dim, a_dim=env.action_dim, a_bound=env.action_bound,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    del env

    for i_subf, h in enumerate(h_list):
        env = Env(max_p=MAX_P, ave_h=h, fix_start=True)
        pos, vel, red_t, yellow_t = av_loop(rl, env, max_ep_step, return_light=True)

        # position statics
        plt.figure(1, figsize=(6, 8))
        if i_subf == 0:
            ax0 = plt.subplot(len(h_list), 1, i_subf+1)
        else:
            plt.subplot(len(h_list), 1, i_subf+1, sharex=ax0, sharey=ax0)

        plt.title('(%s) Traffic flow=%i ($veh/h$)' % (chr(i_subf+97), 3600/h))
        plt.plot(pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Position ($m$)')

        # average speed statics
        plt.figure(2, figsize=(6, 8))
        if i_subf == 0:
            ax1 = plt.subplot(len(h_list), 1, i_subf+1)
        else:
            plt.subplot(len(h_list), 1, i_subf+1, sharex=ax1, sharey=ax1)
        mean_v = np.nanmean(vel, axis=0)*3.6
        plt.title('(%s) Traffic flow=%i ($veh/h$)' % (chr(i_subf + 97), 3600 / h))
        plt.plot(mean_v, c='k')
        red_t = np.array(red_t)
        yellow_t = np.array(yellow_t)
        green_t = np.isnan(yellow_t) & np.isnan(red_t)
        red_t[~np.isnan(red_t)] = mean_v.min()
        yellow_t[~np.isnan(yellow_t)] = mean_v.min()
        green_t = np.where(green_t, mean_v.min(), np.nan)
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(green_t)), green_t, c='g', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Average speed ($km/h$)')

        # average headway
        # plt.figure(3, figsize=(6, 8))
        # if i_subf == 0:
        #     ax2 = plt.subplot(len(h_list), 1, i_subf + 1)
        # else:
        #     plt.subplot(len(h_list), 1, i_subf + 1, sharex=ax2, sharey=ax2)
        # mean_v = np.nanmean(vel, axis=0) * 3.6
        # plt.title('(%s) Traffic flow=%i ($veh/h$)' % (chr(i_subf + 97), 3600 / h))
        # plt.plot(mean_v, c='k')
        # plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        # plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        # plt.plot(np.arange(len(green_t)), green_t, c='g', lw=LIGHT_LW, solid_capstyle="butt")
        # plt.ylabel('Average speed ($km/h$)')

    plt.figure(1, figsize=(6, 8))   # position statics
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, max_ep_step, 500), (np.arange(0, max_ep_step, 500) / 10).astype(np.int32))
    plt.xlim(xmin=0, xmax=max_ep_step)
    plt.ylim(ymin=0)
    plt.tight_layout()

    plt.figure(2, figsize=(6, 8))  # average speed statics
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, max_ep_step, 500), (np.arange(0, max_ep_step, 500) / 10).astype(np.int32))
    plt.xlim(xmin=0, xmax=max_ep_step)
    plt.tight_layout()
    plt.show()


def av_diff_light_duration(l_duration, max_ep_step):
    env = Env(max_p=MAX_P, ave_h=2, fix_start=True)
    rl = DDPG(
        s_dim=env.state_dim, a_dim=env.action_dim, a_bound=env.action_bound,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)

    plt.figure(1, figsize=(6, 8))
    for i_subf, r_dur in enumerate(l_duration):
        l_dur = {'red': r_dur, 'green': r_dur, 'yellow': env.light_duration['yellow']}
        env.light_duration = l_dur
        pos, vel, red_t, yellow_t = av_loop(rl, env, max_ep_step, return_light=True)
        if i_subf == 0:
            ax0 = plt.subplot(len(l_duration), 1, i_subf + 1)
        else:
            plt.subplot(len(l_duration), 1, i_subf + 1, sharex=ax0, sharey=ax0)

        plt.title('(%s) Red=%i$s$; Green=%i$s$; Yellow=%i$s$' %
                  (chr(i_subf + 97), l_dur['red'], l_dur['green'], l_dur['yellow']))
        plt.plot(pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, max_ep_step, 500), (np.arange(0, max_ep_step, 500) / 10).astype(np.int32))
    plt.xlim(xmin=0, xmax=max_ep_step)
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.show()


def plot_reward(parent_path):
    data = []
    for path, _, files in os.walk(parent_path):
        for f in files:
            if f.endswith('.npy'):
                data.append(np.load(path+'/'+f))
    rewards = np.vstack(data)
    r_mean = np.mean(rewards, axis=0)
    r_std = np.std(rewards, axis=0)
    plt.fill_between(np.arange(len(r_mean)), r_mean+r_std, r_mean-r_std, facecolor='b', alpha=0.5)
    plt.plot(r_mean, c='b')
    # plt.errorbar(np.arange(len(r_mean)), r_mean, yerr=r_std, errorevery=50,
    #              capsize=5)
    plt.ylabel('Moving episode-reward')
    plt.xlabel('Episode')
    plt.show()

def plot_mix_traffic():
    pass

if __name__ == '__main__':
    TRAJ_LW = 0.5
    LIGHT_LW = 5
    MAX_P = 1300
    MODEL_DIR = './tf_models/0'
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        if len(sys.argv) == 3:
            MODEL_DIR = MODEL_DIR[:-1] + sys.argv[2]
    else:
        n = 1

    config = [
        dict(fun=plot_av_mv,                # 0
             kwargs={'max_ep_step': 3000, }),
        dict(fun=plot_av_diff_h,            # 1
             kwargs={'h_list': [6., 3., 1.5], 'max_ep_step': 3000, }),
        dict(fun=av_diff_light_duration,    # 2
             kwargs={'max_ep_step': 3000, 'l_duration': [35, 30, 25]}),
        dict(fun=plot_reward,               # 3
             kwargs={'parent_path': './tf_models'})
    ][n]

    config['fun'](**config['kwargs'])