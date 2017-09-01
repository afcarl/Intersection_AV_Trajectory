from env import Env
from tfDDPG import DDPG, DDPGPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize
import sys, os, time

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


def av_loop(RL, env, return_light=True):
    t = 0
    pos = np.zeros((300, MAX_EP_STEP))
    pos[:] = pos.fill(np.nan)
    vel = pos.copy()
    acc = pos.copy()
    if return_light:
        red_t, yellow_t = [], []
    while t < MAX_EP_STEP:
        s = env.reset()
        while True:
            # env.render()
            a = RL.choose_action(s)

            envp, envid, envv = env.car_info['p'][:env.ncs], env.car_info['id'][:env.ncs], env.car_info['v'][:env.ncs]
            pos[envid, t] = envp
            vel[envid, t] = envv
            acc[envid, t] = np.where(envv <= 0, 0., a.ravel())

            s_, r, done, new_s = env.step(a.ravel())
            s = new_s

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
            if done or t == MAX_EP_STEP:
                break

    if return_light:
        return pos, vel, acc, red_t, yellow_t
    else:
        return pos, vel, acc


def mv_loop(env):
    pos = np.zeros((300, MAX_EP_STEP))
    pos[:] = pos.fill(np.nan)
    vel = pos.copy()
    acc = pos.copy()
    t = 0
    while t < MAX_EP_STEP:
        s = env.reset()
        while True:
            # env.render()
            # dx_norm( / 100m), dv_norm (/10), v_norm (/self.max_v), d2l (/1000)
            dx, dv, v, d2l = s[:, -3] * 100., s[:, -2] * 10., s[:, -1] * env.max_v, s[:, 2] * 1000
            if env.is_red_light:
                dv = np.where(dx > d2l, v, dv)
                np.minimum(dx, d2l, out=dx)
            a = IDM(v, dv, dx)

            envp, envid, envv = env.car_info['p'][:env.ncs], env.car_info['id'][:env.ncs], env.car_info['v'][:env.ncs]
            pos[envid, t] = envp
            vel[envid, t] = envv
            acc[envid, t] = np.where(envv <= 0, [0], a)

            s_, r, done, new_s = env.step(a)
            s = new_s
            t += 1
            if done or t == MAX_EP_STEP:
                break
    return pos, vel, acc


def plot_av_mv():
    ave_h = 2.
    env = Env(max_p=MAX_P, ave_h=ave_h, fix_start=True)
    env.set_fps(1000)
    rl = DDPG(
        s_dim=env.state_dim, a_dim=env.action_dim, a_bound=env.action_bound,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    # av test
    av_pos, av_vel, av_acc, red_t, yellow_t = av_loop(rl, env, return_light=True)
    # mv test
    mv_pos, mv_vel, mv_acc = mv_loop(env)

    # position
    plt.figure(1, figsize=(18, 8))
    ax1 = plt.subplot(241)
    plt.title('(a) MVs benchmark')
    plt.plot(mv_pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')

    plt.subplot(245, sharex=ax1, sharey=ax1)
    plt.title('(b) AVs case')
    plt.plot(av_pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500)/10).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    # plt.tight_layout()

    # speed
    for i_subp, vel in enumerate([mv_vel, av_vel], start=1):
        if i_subp == 1:
            ax1 = plt.subplot(2, 4, 2)
        else:
            plt.subplot(2, 4, 6, sharex=ax1)
        mean_v = np.nanmean(vel, axis=0) * 3.6
        plt.title('(%s) %s' % (chr(i_subp + 98), ['MVs benchmark', 'AVs case'][i_subp-1]))
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
        # plt.yticks(rotation=45)
        plt.ylim((0, 110))
        plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) / 10).astype(np.int32))
    # plt.tight_layout()

    # time gap
    for i_subp, (vel, pos) in enumerate(zip([mv_vel, av_vel], [mv_pos, av_pos]), start=1):
        if i_subp == 1:
            ax1 = plt.subplot(2, 4, 3)
        else:
            plt.subplot(2, 4, 7, sharex=ax1)
        dx = -np.diff(pos, axis=0)
        time_gap = dx / (vel[1:] + 1e-3)
        min_t_gap = np.nanmin(time_gap, axis=0)
        plt.title('(%s) %s' % (chr(i_subp + 100), ['MVs benchmark', 'AVs case'][i_subp - 1]))
        plt.plot(min_t_gap, c='k')
        red_t = np.array(red_t)
        yellow_t = np.array(yellow_t)
        green_t = np.isnan(yellow_t) & np.isnan(red_t)
        red_t[~np.isnan(red_t)] = np.nanmin(min_t_gap)
        yellow_t[~np.isnan(yellow_t)] = np.nanmin(min_t_gap)
        green_t = np.where(green_t, np.nanmin(min_t_gap), np.nan)
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(green_t)), green_t, c='g', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Minimum time gap ($km/h$)')
        plt.ylim((0.5, 2.5))
        plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) / 10).astype(np.int32))
    # plt.tight_layout()

    # acceleration
    for i_subp, acc in enumerate([mv_acc, av_acc], start=1):
        if i_subp == 1:
            ax1 = plt.subplot(2, 4, 4)
        else:
            plt.subplot(2, 4, 8, sharex=ax1)
        std_acc = np.nanstd(acc, axis=0)
        plt.title('(%s) %s' % (chr(i_subp + 102), ['MVs benchmark', 'AVs case'][i_subp - 1]))
        plt.plot(std_acc, c='k')
        red_t = np.array(red_t)
        yellow_t = np.array(yellow_t)
        green_t = np.isnan(yellow_t) & np.isnan(red_t)
        red_t[~np.isnan(red_t)] = np.nanmin(std_acc)
        yellow_t[~np.isnan(yellow_t)] = np.nanmin(std_acc)
        green_t = np.where(green_t, np.nanmin(std_acc), np.nan)
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(green_t)), green_t, c='g', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Standard deviation of acceleration')
        plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) / 10).astype(np.int32))
        plt.ylim((0, 4))
    plt.tight_layout()
    plt.show()


def plot_av_diff_h(h_list):
    env = Env(max_p=MAX_P, ave_h=2, fix_start=True)
    rl = DDPG(
        s_dim=env.state_dim, a_dim=env.action_dim, a_bound=env.action_bound,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    del env

    for i_subf, h in enumerate(h_list, start=1):
        env = Env(max_p=MAX_P, ave_h=h, fix_start=True)
        pos, vel, acc, red_t, yellow_t = av_loop(rl, env, return_light=True)

        # position statics
        plt.figure(1, figsize=(6, 8))
        if i_subf == 1:
            ax0 = plt.subplot(len(h_list), 1, i_subf)
        else:
            plt.subplot(len(h_list), 1, i_subf, sharex=ax0, sharey=ax0)

        plt.title('(%s) Traffic flow=%i ($veh/h$)' % (chr(i_subf+96), 3600/h))
        plt.plot(pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Position ($m$)')

        # average speed statics
        plt.figure(2, figsize=(6, 8))
        if i_subf == 1:
            ax1 = plt.subplot(len(h_list), 1, i_subf)
        else:
            plt.subplot(len(h_list), 1, i_subf, sharex=ax1, sharey=ax1)
        mean_v = np.nanmean(vel, axis=0)*3.6
        plt.title('(%s) Traffic flow=%i ($veh/h$)' % (chr(i_subf + 96), 3600 / h))
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
        # if i_subf == 1:
        #     ax2 = plt.subplot(len(h_list), 1, i_subf)
        # else:
        #     plt.subplot(len(h_list), 1, i_subf, sharex=ax2, sharey=ax2)
        # mean_v = np.nanmean(vel, axis=0) * 3.6
        # plt.title('(%s) Traffic flow=%i ($veh/h$)' % (chr(i_subf + 96), 3600 / h))
        # plt.plot(mean_v, c='k')
        # plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        # plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        # plt.plot(np.arange(len(green_t)), green_t, c='g', lw=LIGHT_LW, solid_capstyle="butt")
        # plt.ylabel('Average speed ($km/h$)')

    plt.figure(1, figsize=(6, 8))   # position statics
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) / 10).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.tight_layout()

    plt.figure(2, figsize=(6, 8))  # average speed statics
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) / 10).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.tight_layout()
    plt.show()


def av_diff_light_duration(l_duration):
    env = Env(max_p=MAX_P, ave_h=2, fix_start=True)
    rl = DDPG(
        s_dim=env.state_dim, a_dim=env.action_dim, a_bound=env.action_bound,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)

    plt.figure(1, figsize=(6, 8))
    for i_subf, r_dur in enumerate(l_duration, start=1):
        l_dur = {'red': r_dur, 'green': r_dur, 'yellow': env.light_duration['yellow']}
        env.light_duration = l_dur
        pos, vel, acc, red_t, yellow_t = av_loop(rl, env, return_light=True)
        if i_subf == 1:
            ax0 = plt.subplot(len(l_duration), 1, i_subf)
        else:
            plt.subplot(len(l_duration), 1, i_subf, sharex=ax0, sharey=ax0)

        plt.title('(%s) Red=%i$s$; Green=%i$s$; Yellow=%i$s$' %
                  (chr(i_subf + 96), l_dur['red'], l_dur['green'], l_dur['yellow']))
        plt.plot(pos.T, c='k', alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) / 10).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
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


def plot_mix_traffic(av_rate):
    env = Env(max_p=MAX_P, ave_h=2, fix_start=True)
    rl = DDPG(
        s_dim=env.state_dim, a_dim=env.action_dim, a_bound=env.action_bound,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    plt.figure(1, figsize=(6, 8))
    for i_subp, rate in enumerate(av_rate, start=1):
        av_mask = np.random.choice([True, False], 300, replace=True, p=[rate, 1-rate])

        # av_mask = np.zeros((300, ), dtype=np.bool)
        # av_mask[::10] = True

        pos = np.zeros((300, MAX_EP_STEP))
        pos[:] = pos.fill(np.nan)
        vel = pos.copy()
        t = 0
        red_t, yellow_t = [], []
        while t < MAX_EP_STEP:
            s = env.reset()
            while True:
                am = av_mask[:env.ncs]
                a = np.empty_like(am, dtype=np.float32)
                # dx_norm( / 100m), dv_norm (/10), v_norm (/self.max_v), d2l (/1000)
                if np.any(~am):
                    dx, dv, v, d2l = s[~am, -3] * 100., s[~am, -2] * 10., s[~am, -1] * env.max_v, s[~am, 2] * 1000
                    if env.is_red_light:
                        dv = np.where(dx > d2l, v, dv)
                        np.minimum(dx, d2l, out=dx)
                    a[~am] = IDM(v, dv, dx)
                if np.any(am):
                    a[am] = rl.choose_action(s[am]).ravel()

                s_, r, done, new_s = env.step(a)
                s = new_s
                envp, envid, envv = env.car_info['p'][:env.ncs], env.car_info['id'][:env.ncs], env.car_info['v'][
                                                                                               :env.ncs]
                pos[envid, t] = envp
                vel[envid, t] = envv
                if env.is_red_light and (env.t_light >= env.light_duration['yellow']):
                    red_t.append(env.light_p)
                else:
                    red_t.append(np.nan)
                if env.is_red_light and (env.t_light < env.light_duration['yellow']):
                    yellow_t.append(env.light_p)
                else:
                    yellow_t.append(np.nan)
                t += 1
                if done or t == MAX_EP_STEP:
                    break
        if i_subp == 1:
            ax0 = plt.subplot(len(av_rate), 1, i_subp)
        else:
            plt.subplot(len(av_rate), 1, i_subp, sharex=ax0, sharey=ax0)

        plt.title('({0}) {1:.0f}% AVs'.format(chr(i_subp + 96), rate*100))
        for icar, traj in enumerate(pos):
            c = 'r' if av_mask[icar] else 'b'
            plt.plot(traj, c=c, alpha=0.8, lw=TRAJ_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) / 10).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    TRAJ_LW = 0.5
    LIGHT_LW = 5
    MAX_P = 1200
    MODEL_DIR = './tf_models/0'
    MAX_EP_STEP = 2500
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        if len(sys.argv) == 3:
            MODEL_DIR = MODEL_DIR[:-1] + sys.argv[2]
    else:
        n = 0

    config = [
        dict(fun=plot_av_mv,                # 0
             kwargs={}),
        dict(fun=plot_av_diff_h,            # 1
             kwargs={'h_list': [6., 3., 1.5],}),
        dict(fun=av_diff_light_duration,    # 2
             kwargs={'l_duration': [35, 30, 25]}),
        dict(fun=plot_reward,               # 3
             kwargs={'parent_path': './tf_models'}),
        dict(fun=plot_mix_traffic,          # 4
             kwargs={'av_rate': np.arange(0, 1.1, 0.25)}),
    ][n]

    t0 = time.time()
    config['fun'](**config['kwargs'])
    print(time.time()-t0)