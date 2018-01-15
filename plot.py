from env import Env
from tfDDPG import DDPG, DDPGPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize, njit
import os, time, argparse
from ColorLine import colorline


ENV = Env()
TRAJ_LW = 0.4
LIGHT_LW = 3
LIGHT_P = 1500
DEFAULT_H = 4.
MAX_EP_STEP = int(380 / ENV.dt)
STATE_DIM = ENV.state_dim
ACTION_DIM = ENV.action_dim
ACTION_BOUND = ENV.action_bound


Kij = np.array([
    [   # positive acceleration
        [-7.735, 0.2295, -5.61e-3, 9.773e-5],
        [0.02799, 0.0068, -7.722E-4, 8.38E-6],
        [-2.228E-4, -4.402E-5, 7.90E-7, 8.17E-7],
        [1.09E-6, 4.80E-8, 3.27E-8, -7.79E-9],
    ],
    [   # negative acceleration
        [-7.73, -0.01799, -4.27E-3, 1.8829E-4],
        [0.02804, 7.72E-3, 8.375E-4, 3.387E-5],
        [-2.199E-4, -5.219E-5, -7.44E-6, 2.77E-7],
        [1.08E-6, 2.47E-7, 4.87E-8, 3.79E-10],
    ],
])


def iTTC(vel, pos, car_len):
    ittc = np.empty((vel.shape[0]-1, ), dtype=np.float32)
    for i, (vl, pl, vf, pf) in enumerate(zip(vel[:-1], pos[:-1], vel[1:], pos[1:])):
        dv = vf - vl
        dx = pl - pf - car_len
        dx = dx[~np.isnan(dv)]
        dv = dv[~np.isnan(dv)]
        ittc[i] = np.sum(np.maximum(dv / dx, 0))
    return ittc


def color_loop(pos, vel):
    for i in range(pos.shape[0]):
        if np.all(np.isnan(pos[i, :])):
            break
        not_nan_idx = (~np.isnan(pos[i])) & (~np.isnan(vel[i]))
        pos_, vel_ = pos[i, not_nan_idx], vel[i, not_nan_idx]
        colorline(np.arange(pos.shape[1])[not_nan_idx], pos_, (1-vel_ / ENV.max_v)/2,
                  linewidth=TRAJ_LW, cmap='brg', alpha=0.7)

@njit
def emission_t(acc, vel, e, Kij):
    for t in range(acc.shape[1]):
        at, vt = acc[:, t], vel[:, t]
        if np.all(np.isnan(at)):
            continue
        else:
            e_tmp = 0.
            for a, v in zip(at, vt):  # time t
                if np.isnan(a):
                    continue
                else:
                    tmp = 0.
                    k = Kij[0] if a > 0 else Kij[1]
                    for i in range(4):
                        for j in range(4):
                            if a < -10.:
                                a = -10.  # m/s^2 lower bound
                            elif a > 2.:
                                a = 2.  # m/s^2 upper bound
                            if v < 0.:
                                v = 0.  # m/s lower bound
                            elif v > 36.:
                                v = 36.  # m/s upper bound

                            tmp += k[i, j] * v**i * a**j    # l/s
                    e_tmp += np.exp(tmp)/10     # /10 for average in 1 second
            e[t] = e_tmp
    return e


@njit
def emission_car(acc, vel, Kij):
    e = np.zeros((acc.shape[0],), dtype=np.float32)
    for n in range(acc.shape[0]):
        an, vn = acc[n], vel[n]
        e_tmp = 0.
        mask = (~np.isnan(an)) & (~np.isnan(vn))
        an = an[mask]
        vn = vn[mask]
        for a, v in zip(an, vn):  # time t
            tmp = 0.
            k = Kij[0] if a > 0 else Kij[1]
            for i in range(4):
                for j in range(4):
                    # a *= 3.6
                    # v *= 3.6
                    # if a < -5.4: a = -5.4  # m/s^2 lower bound
                    # elif a > 13.3: a = 13.3  # m/s^2 upper bound
                    # if v < 0.:
                    #     v = 0.  # m/s lower bound
                    # elif v > 120.:
                    #     v = 120.

                    if a < -10.: a = -10.           # m/s^2 lower bound
                    elif a > 2.: a = 2.         # m/s^2 upper bound
                    if v < 0.: v = 0.             # m/s lower bound
                    elif v > 36.: v = 36.         # m/s upper bound

                    tmp += k[i, j] * v**i * a**j    # l/s
            e_tmp += np.exp(tmp)/10     # /10 for average in 1 second
        e[n] = e_tmp
    return e


@njit
def travel_time_car(pos):
    travel_time = np.zeros((pos.shape[0], ), dtype=np.float32)
    for n in range(pos.shape[0]):
        pos_n = pos[n]
        valid_time = pos_n[~np.isnan(pos_n)].shape[0]
        travel_time[n] = valid_time * ENV.dt   # second
    return travel_time


@vectorize('float32(float32, float32, float32)')
def IDM(vn, delta_vn, delta_xn):
    v0 = 120/3.6
    T_desired = 1.6
    a_max = 0.73
    b_max = 1.67
    s0 = 2.
    value = vn*T_desired+vn*delta_vn/(2*(a_max*b_max)**(1/2))
    s_star = s0 + np.maximum(value, 0.)
    a = a_max*(1.-(vn/v0)**4-(s_star/(delta_xn+1e-4))**2)
    return a


def av_loop(RL, env, return_light=False):
    t = 0
    pos = np.zeros((300, MAX_EP_STEP), dtype=np.float32)
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


def mv_loop(env, return_light=False):
    fdx = env.feature_info["dx"]
    fdv = env.feature_info["dv"]
    fv = env.feature_info["v"]
    fd2l = env.feature_info["d2l"]

    pos = np.zeros((300, MAX_EP_STEP), dtype=np.float32)
    pos[:] = pos.fill(np.nan)
    vel = pos.copy()
    acc = pos.copy()
    t = 0
    if return_light:
        red_t, yellow_t = [], []
    while t < MAX_EP_STEP:
        s = env.reset()
        while True:
            # env.render()
            # dx_norm( / 100m), dv_norm (/10), v_norm (/self.max_v), d2l (/1000)
            dx, dv, v, d2l = s[:, fdx["i"]] / fdx["f"], s[:, fdv["i"]] / fdv["f"], s[:, fv["i"]] / fv["f"], s[:, fd2l["i"]] / fd2l["f"]
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


def plot_av_mv(ave_h):
    rl = DDPG(
        s_dim=STATE_DIM, a_dim=ACTION_DIM, a_bound=ACTION_BOUND,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    rl.reset()

    env = Env(light_p=LIGHT_P, ave_h=ave_h, fix_start=True)
    env.set_fps(1000)

    # av test
    av_pos, av_vel, av_acc, red_t, yellow_t = av_loop(rl, env, return_light=True)
    # mv test
    mv_pos, mv_vel, mv_acc = mv_loop(env)

    # emission and iTTC
    nec = 45  # number of vehicles for emission test
    mv_e = emission_car(mv_acc[:nec], mv_vel[:nec],  Kij)
    av_e = emission_car(av_acc[:nec], av_vel[:nec],  Kij)

    ittc_av = iTTC(av_vel[:nec], av_pos[:nec], env.car_l)
    ittc_mv = iTTC(mv_vel[:nec], mv_pos[:nec], env.car_l)

    fig1 = plt.figure(1, figsize=(8, 6))

    # trajectory mv
    ax1 = plt.subplot(221)
    plt.title('(a) First %i MVs' % nec)
    color_loop(mv_pos[:nec], mv_vel[:nec])
    # plt.plot(mv_pos[:nec].T, c='k', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')
    plt.text(x=MAX_EP_STEP*.95, y=100, s='Average emission=%.2f ($l/car$)\nAverage iTTC=%.2f$/car$' %
            (np.nansum(mv_e[:nec])/nec, np.nanmean(ittc_mv[:nec])), ha="right", fontdict={'size': 8},
             bbox=dict(facecolor='white', alpha=0.9))

    # trajectory av
    plt.subplot(223, sharex=ax1, sharey=ax1)
    plt.title('(b) First %i AVs' % nec)
    color_loop(av_pos[:nec], av_vel[:nec])
    # plt.plot(av_pos[:nec].T, c='k', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.text(x=MAX_EP_STEP*.95, y=100, s='Average emission=%.2f ($l/car$)\nAverage iTTC=%.2f$/car$' %
             (np.nansum(av_e[:nec])/nec, np.nanmean(ittc_av[:nec])), ha='right', fontdict={'size': 8,},
             bbox=dict(facecolor='white', alpha=0.9))

    # emission
    plt.subplot(222)
    plt.scatter(np.arange(nec), mv_e, s=20, c='b', alpha=0.5, label='MVs')
    plt.scatter(np.arange(nec), av_e, s=20, c='r', alpha=0.5, label='AVs')
    plt.legend(loc='best')
    plt.xlim((0, nec))
    plt.ylabel('Total emission ($liter$)')
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(c) Emission (VT-Micro model)')

    # iTTC
    plt.subplot(224)
    plt.scatter(np.arange(1, len(ittc_mv) + 1), ittc_mv, s=20, c='b', alpha=0.5, label='MVs')
    plt.scatter(np.arange(1, len(ittc_mv) + 1), ittc_av, s=20, c='r', alpha=0.5, label='AVs')
    plt.legend(loc='best')
    plt.ylabel('iTTC')
    plt.xlim((0, nec))
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(d) Safety')
    plt.tight_layout()

    # position
    fig2 = plt.figure(2, figsize=(18, 6))
    ax1 = plt.subplot(241)
    plt.title('(a) MVs benchmark')
    # plt.plot(mv_pos.T, c='k', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
    mv_pos = mv_pos[:, ~np.all(np.isnan(mv_pos), axis=0)]
    mv_vel = mv_vel[:, ~np.all(np.isnan(mv_vel), axis=0)]
    color_loop(mv_pos, mv_vel)
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')

    plt.subplot(245, sharex=ax1, sharey=ax1)
    plt.title('(b) AVs case')
    # plt.plot(av_pos.T, c='k', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
    av_pos = av_pos[:, ~np.all(np.isnan(av_pos), axis=0)]
    av_vel = av_vel[:, ~np.all(np.isnan(av_vel), axis=0)]
    color_loop(av_pos, av_vel)
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
        plt.xlim((0, MAX_EP_STEP))
        plt.ylim((0, 110))
        plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    # plt.tight_layout()

    # time gap
    for i_subp, (vel, pos) in enumerate(zip([mv_vel, av_vel], [mv_pos, av_pos]), start=1):
        if i_subp == 1:
            ax1 = plt.subplot(2, 4, 3)
        else:
            plt.subplot(2, 4, 7, sharex=ax1)
        dx = -np.diff(pos, axis=0)
        time_gap = dx / (vel[1:] + 1e-3)
        time_gap = time_gap[:, ~np.all(np.isnan(time_gap), axis=0)]
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
        plt.ylabel('Minimum time gap ($s$)')
        plt.ylim((0., 2.5))
        plt.xlim((0, MAX_EP_STEP))
        plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
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
        plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
        plt.ylim((0, 4))
        plt.xlim((0, MAX_EP_STEP))
    plt.tight_layout()
    if SAVE_FIG:
        fig1.savefig('./results/fuel_safe%i.png' % model_n, format='png', dpi=700)
        fig2.savefig('./results/pvh%i.png' % model_n, format='png', dpi=700)
        plt.close(fig1)
        plt.close(fig2)
        print('saved fuel_safe.png, pvh.png')
    else:
        plt.show()


def plot_av_diff_h(h_list):
    rl = DDPG(
        s_dim=STATE_DIM, a_dim=ACTION_DIM, a_bound=ACTION_BOUND,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    rl.reset()
    env = Env(light_p=LIGHT_P, ave_h=3., fix_start=True)
    del env

    for i_subf, h in enumerate(h_list, start=1):
        env = Env(light_p=LIGHT_P, ave_h=h, fix_start=True)
        pos_av, vel_av, acc_av, red_t, yellow_t = av_loop(rl, env, return_light=True)
        # pos_mv, vel_mv, acc_mv = mv_loop(env)

        # position statics
        plt.figure(1, figsize=(6, 8))
        if i_subf == 1:
            ax0 = plt.subplot(len(h_list), 1, i_subf)
        else:
            plt.subplot(len(h_list), 1, i_subf, sharex=ax0, sharey=ax0)

        plt.title('(%s) Traffic flow=%i ($veh/h$)' % (chr(i_subf+96), 3600/h))
        # plt.plot(pos_av.T, c='k', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
        color_loop(pos_av, vel_av)
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Position ($m$)')

        # average speed statics
        plt.figure(2, figsize=(6, 8))
        if i_subf == 1:
            ax1 = plt.subplot(len(h_list), 1, i_subf)
        else:
            plt.subplot(len(h_list), 1, i_subf, sharex=ax1, sharey=ax1)
        mean_v = np.nanmean(vel_av, axis=0)*3.6
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

    fig1 = plt.figure(1)   # position statics
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.tight_layout()

    fig2 = plt.figure(2)  # average speed statics
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.tight_layout()
    if SAVE_FIG:
        fig1.savefig('./results/headways_p%i.png' % model_n, format='png', dpi=500)
        fig2.savefig('./results/headways_v%i.png' % model_n, format='png', dpi=500)
        plt.close(fig1)
        plt.close(fig2)
        print('saved headways_p, headways_v')
    else:
        plt.show()


def av_diff_light_duration(l_duration):
    rl = DDPG(
        s_dim=STATE_DIM, a_dim=ACTION_DIM, a_bound=ACTION_BOUND,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    rl.reset()
    env = Env(light_p=LIGHT_P, ave_h=DEFAULT_H, fix_start=True)

    fig = plt.figure(1, figsize=(6, 8))
    for i_subf, r_dur in enumerate(l_duration, start=1):
        l_dur = {'red': r_dur, 'green': r_dur, 'yellow': env.light_duration['yellow']}
        env.light_duration = l_dur
        pos, vel, acc, red_t, yellow_t = av_loop(rl, env, return_light=True)
        if i_subf == 1:
            ax0 = plt.subplot(len(l_duration), 1, i_subf)
        else:
            plt.subplot(len(l_duration), 1, i_subf, sharex=ax0, sharey=ax0)

        plt.title('(%s) Green=%i$s$; Yellow=%i$s$; Red=%i$s$' %
                  (chr(i_subf + 96), l_dur['green'], l_dur['yellow'], l_dur['red']))
        # plt.plot(pos.T, c='k', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
        color_loop(pos, vel)
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.tight_layout()
    if SAVE_FIG:
        fig.savefig('./results/diff_light_duration%i.png' % model_n, format='png', dpi=500)
        plt.close(fig)
        print('saved diff_light_duration')
    else:
        plt.show()


def plot_reward(download, n_models):
    import subprocess, signal, requests, shutil
    import pandas as pd
    tags = ['a_loss%2Factor_loss', 'c_loss%2Fcritic_loss', 'ep_reward']

    if os.path.exists('./results/tensorboard/') and download:
        shutil.rmtree('./results/tensorboard/')
    if download:
        pro = subprocess.Popen(["tensorboard", "--logdir", "log"])
        os.makedirs('./results/tensorboard/', exist_ok=True)
        for i in range(n_models):
            for tag in tags:
                response = requests.get('http://localhost:6006/data/plugin/scalars/scalars?run={}&tag={}&format=csv'.format(
                    i, tag
                ))
                print(response.url)
                data = response.content
                with open('./results/tensorboard/%i-%s.csv' % (i, tag), 'wb') as f:
                    f.write(data)
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

    def moving_average(a, alpha):
        res = np.empty_like(a, dtype=float)
        for i in range(len(res)):
            if i == 0:
                res[i] = a[i]
            else:
                res[i] = alpha * res[i-1] + (1.-alpha) * a[i]
        return res

    for i in range(n_models):
        ep_reward = pd.read_csv('./results/tensorboard/{}-ep_reward.csv'.format(i))
        y = moving_average(ep_reward['Value'], alpha=0.6) * float(MAX_EP_STEP)
        plt.plot(ep_reward['Step'], y)
    plt.ylabel('Episode reward')
    plt.xlabel('Learning step')
    plt.xlim(xmin=0, xmax=ep_reward['Step'].max())
    if SAVE_FIG:
        plt.savefig('./results/reward.png', format='png', dpi=500)
        print('saved reward')
        plt.close(plt.gcf())
    else:
        plt.show()



def plot_mix_traffic(av_rate):
    rl = DDPG(
        s_dim=STATE_DIM, a_dim=ACTION_DIM, a_bound=ACTION_BOUND,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    rl.reset()
    env = Env(light_p=LIGHT_P, ave_h=DEFAULT_H, fix_start=True)
    fdx = env.feature_info["dx"]
    fdv = env.feature_info["dv"]
    fv = env.feature_info["v"]
    fd2l = env.feature_info["d2l"]

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
                    dx, dv, v, d2l = s[~am, fdx["i"]] / fdx["f"], s[~am, fdv["i"]] / fdv["f"], s[~am, fv["i"]] / fv["f"], s[~am, fd2l["i"]] / fd2l["f"]
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

        for icar, traj in enumerate(pos):
            c = 'r' if av_mask[icar] else 'b'
            plt.plot(traj, c=c, alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        plt.ylabel('({0}) {1:.0f}% AVs'.format(chr(i_subp + 96), rate*100) + '\nPosition ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('./results/mix_traffic%i.png' % model_n, format='png', dpi=500)
        print('saved mix_traffic')
        plt.gcf().close()
    else:
        plt.show()


def plot_demand_change(demand_change):
    env = Env(light_p=LIGHT_P, ave_h=demand_change[0], fix_start=True)
    t = 0
    pos = np.zeros((300, MAX_EP_STEP), dtype=np.float32)
    pos[:] = pos.fill(np.nan)
    vel = pos.copy()
    acc = pos.copy()
    red_t, yellow_t = [], []
    while t < MAX_EP_STEP:
        s = env.reset()
        while True:
            a = rl.choose_action(s)

            envp, envid, envv = env.car_info['p'][:env.ncs], env.car_info['id'][:env.ncs], env.car_info['v'][:env.ncs]
            pos[envid, t] = envp
            vel[envid, t] = envv
            acc[envid, t] = np.where(envv <= 0, [0.], a.ravel())

            s_, r, done, new_s = env.step(a.ravel())
            s = new_s

            if env.is_red_light and (env.t_light >= env.light_duration['yellow']):
                red_t.append(env.light_p)
            else:
                red_t.append(np.nan)
            if env.is_red_light and (env.t_light < env.light_duration['yellow']):
                yellow_t.append(env.light_p)
            else:
                yellow_t.append(np.nan)

            t += 1
            if t < MAX_EP_STEP / 3:
                env.ave_headway = 3600/demand_change[0]
            elif t < MAX_EP_STEP * 2 / 3:
                env.ave_headway = 3600/demand_change[1]
            else:
                env.ave_headway = 3600/demand_change[2]
            if done or t == MAX_EP_STEP:
                break
    plt.title('Traffic demand %i-%i-%i $veh/h$' % (demand_change[0], demand_change[1], demand_change[2]))
    # plt.plot(pos.T, c='k', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
    color_loop(pos, vel)
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    if SAVE_FIG:
        plt.savefig('./results/diff_demand%i.png' % model_n, format='png', dpi=500)
        print('saved diff_demand')
        plt.gcf().close()
    else:
        plt.show()


def plot_xiaobo_demo1(ave_flow, nec):
    rl = DDPG(
        s_dim=STATE_DIM, a_dim=ACTION_DIM, a_bound=ACTION_BOUND,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    rl.reset()
    env = Env(light_p=LIGHT_P, ave_h=3600/ave_flow, fix_start=True)
    env.set_fps(2000)
    # av test
    av_pos, av_vel, av_acc, red_t, yellow_t = av_loop(rl, env, return_light=True)
    # mv test
    mv_pos, mv_vel, mv_acc = mv_loop(env)

    fig = plt.figure(1, figsize=(10, 7))

    # emission for each car
    mv_e = emission_car(mv_acc[nec[0]:nec[1]], mv_vel[nec[0]:nec[1]], Kij)
    av_e = emission_car(av_acc[nec[0]:nec[1]], av_vel[nec[0]:nec[1]], Kij)

    # safety for each car
    ittc_av = iTTC(av_vel[nec[0]:nec[1]], av_pos[nec[0]:nec[1]], env.car_l)
    ittc_mv = iTTC(mv_vel[nec[0]:nec[1]], mv_pos[nec[0]:nec[1]], env.car_l)

    # travel time for each car
    travel_t_mv = travel_time_car(mv_pos[nec[0]:nec[1]])
    travel_t_av = travel_time_car(av_pos[nec[0]:nec[1]])

    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)  # trajectory mv
    plt.title('(a) Traditional vehicles')
    trajs = plt.plot(mv_pos[nec[0]:nec[1]].T, c='r', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.ylabel('Position ($m$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.xlabel('Time ($s$)')

    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)    # trajectory av
    plt.title('(b) CAVs')
    trajs = plt.plot(av_pos[nec[0]:nec[1]].T, c='b', alpha=0.9, lw=TRAJ_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
    plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
    plt.yticks(())
    plt.xlabel('Time ($s$)')
    plt.xticks(np.arange(0, MAX_EP_STEP, 500), (np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
    plt.xlim(xmin=0, xmax=MAX_EP_STEP)
    plt.ylim(ymin=0)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)  # travel time
    plt.scatter(np.arange(nec[0], nec[1]), travel_t_mv, s=20, c='r', alpha=0.5, label='Traditional vehicles')
    plt.scatter(np.arange(nec[0], nec[1]), travel_t_av, s=20, c='b', alpha=0.5, label='CAVs')
    plt.legend(loc='best', prop={'size': 6})
    plt.xlim((nec[0], nec[1]))
    plt.ylabel('Second')
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(c) Travel time')
    plt.text(5-2, 80, 'Red phase 1', color='r', fontsize=8)
    plt.text(11-2, 120, 'Red phase 2', color='r', fontsize=8)
    plt.text(17-2, 160, 'Red phase 3', color='r', fontsize=8)
    plt.text(23-3, 195, 'Red phase 4', color='r', fontsize=8)
    plt.text(14-4, 50, 'Red phase', color='b', fontsize=8)

    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)  # emission
    plt.scatter(np.arange(nec[0], nec[1]), mv_e, s=20, c='r', alpha=0.5, label='Traditional vehicles')
    plt.scatter(np.arange(nec[0], nec[1]), av_e, s=20, c='b', alpha=0.5, label='CAVs')
    plt.legend(loc='best', prop={'size': 6})
    plt.xlim((nec[0], nec[1]))
    plt.ylabel('Liter')
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(d) Emission (VT-Micro model)')
    plt.text(5 - 2, 0.055, 'Red phase 1', color='r', fontsize=8)
    plt.text(11 - 2, 0.07, 'Red phase 2', color='r', fontsize=8)
    plt.text(17 - 2, 0.088, 'Red phase 3', color='r', fontsize=8)
    plt.text(23 - 3, 0.105, 'Red phase 4', color='r', fontsize=8)
    plt.text(14 - 4, 0.04, 'Red phase', color='b', fontsize=8)

    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)  # iTTC
    plt.scatter(np.arange(nec[0]+1, nec[1]), ittc_mv, s=20, c='r', alpha=0.5, label='Traditional vehicles')
    plt.scatter(np.arange(nec[0]+1, nec[1]), ittc_av, s=20, c='b', alpha=0.5, label='CAVs')
    plt.legend(loc='best', prop={'size': 6})
    plt.ylabel('Crash risk')
    plt.xlim((nec[0], nec[1]))
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(e) Safety')
    plt.text(5 - 2, 45, 'Red phase 1', color='r', fontsize=8)
    plt.text(11 - 2, 60, 'Red phase 2', color='r', fontsize=8)
    plt.text(17 - 2, 70, 'Red phase 3', color='r', fontsize=8)
    plt.text(23 - 3, 85, 'Red phase 4', color='r', fontsize=8)
    plt.text(14 - 2, 10, 'Red phase', color='b', fontsize=8)

    plt.tight_layout(pad=0.2)
    if SAVE_FIG:
        plt.savefig('./results/demo_car%i_car%i_%i.png' % (nec[0], nec[1], model_n), format='png', dpi=500)
        plt.gcf().close()
    else:
        plt.show()


def plot_xiaobo_demo2(ave_flow, nec, av_rate, drawing, save):
    from matplotlib import cm
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    if drawing:
        global nec_n, scatters, lines, pos, vel, acc, trajectory_x, trajectory_y

        if av_rate == 0:
            av_idx = [-1]
        elif av_rate == 0.1:
            av_idx = [3,12,20]
        elif av_rate == 0.5:
            av_idx = [0,3,4,5,9,10,11,15,19,20,21,22,25,28]
        fig = plt.figure(1, figsize=(10, 7))

        env = Env(light_p=LIGHT_P, ave_h=3600 / ave_flow, fix_start=True)
        env.set_fps(2000)
        plt.xlim(xmin=0, xmax=MAX_EP_STEP)
        plt.ylim(ymin=0, ymax=env.max_p+50)
        nec_n = 0
        pos = np.zeros((300, MAX_EP_STEP), dtype=np.float32)
        pos[:] = pos.fill(np.nan)
        vel = pos.copy()
        acc = pos.copy()
        ave_h = 3600 / ave_flow
        gd, yd, rd = env.light_duration['green'] * 10, env.light_duration['yellow'] * 10, env.light_duration['red'] * 10
        init_v = env.max_v
        n_colors = 15
        g = np.linspace(0, 1, n_colors)[::-1]
        x = np.linspace(0, MAX_EP_STEP, MAX_EP_STEP)
        fills = list(range(n_colors))[::-1]

        trajectory_x = []
        trajectory_y = []
        scatters = []
        lines = []

        def smooth(x, window_len=11, window='hanning'):
            if window_len < 3:
                return x

            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.' + window + '(window_len)')

            y = np.convolve(w / w.sum(), s, mode='valid')
            # return y
            return y[window_len//2:y.size-window_len//2+1]

        def onclick(event):
            # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #       (event.button, event.x, event.y, event.xdata, event.ydata))
            if event.button == 1:
                scatters.append(plt.scatter(event.xdata, event.ydata, c='k'))
                trajectory_x.append(event.xdata)
                trajectory_y.append(event.ydata)
                if len(trajectory_x) > 1:
                    lines.append(plt.plot(trajectory_x, trajectory_y, c='b')[0])
            fig.canvas.draw()

        def onkey(event):
            global nec_n, lines, scatters, pos, vel, acc, trajectory_x, trajectory_y
            if event.key == 'backspace':
                if len(trajectory_x) > 0:
                    if len(trajectory_x) > 1:
                        lines[-1].remove()
                        lines.pop()
                    scatters[-1].remove()
                    scatters.pop()
                    trajectory_x.pop()
                    trajectory_y.pop()
            elif event.key == 'enter':
                [s.remove() for s in scatters]
                [l.remove() for l in lines]
                init_t = int(nec_n * ave_h * 10)
                xvals = np.arange(init_t, int(trajectory_x[-1]), dtype=np.float32)
                yinterp = np.interp(xvals, trajectory_x, trajectory_y)
                yma = smooth(yinterp, 12)
                plt.plot(xvals, yma, c='b')

                pos[nec_n, init_t: int(trajectory_x[-1])] = yma
                v = np.diff(yma)/env.dt
                v = np.concatenate((v[0:1], v))
                vel[nec_n, init_t: int(trajectory_x[-1])] = v
                a = np.diff(v)/env.dt
                acc[nec_n, init_t: int(trajectory_x[-1])] = np.concatenate(([0.], a))

                trajectory_x = []
                trajectory_y = []
                lines, scatters = [], []
                nec_n += 1

            elif event.key == ' ':      # generate mv
                if nec_n <= nec:
                    # car info
                    # initial condition
                    init_t = int(nec_n * ave_h * 10)
                    pos[nec_n, init_t] = 0.
                    vel[nec_n, init_t] = init_v
                    acc[nec_n, init_t] = 0.
                    if nec_n not in av_idx:
                        for t in range(init_t + 1, MAX_EP_STEP):  # t for car showing in the figure
                            # check in red phase
                            t_in_cycle = t % (gd + yd + rd)
                            if t_in_cycle < gd:
                                is_red_light = False
                            else:
                                is_red_light = True

                            # driving info
                            if nec_n == 0 or np.isnan(pos[nec_n - 1, t]):
                                dv = np.array([0., ], dtype=np.float32)  # follower - leader
                                dx = np.array([100., ], dtype=np.float32)  # leader - follower
                            else:
                                dv = vel[nec_n, t - 1: t] - vel[nec_n - 1, t - 1: t]
                                dx = pos[nec_n - 1, t - 1: t] - pos[nec_n, t - 1: t]
                            v = vel[nec_n, t - 1: t]
                            dx -= env.car_l

                            if is_red_light:
                                d2l = LIGHT_P - pos[nec_n, t - 1: t]
                                dv = np.where(dx > d2l, v, dv)
                                np.minimum(dx, d2l, out=dx)

                            a = IDM(v, dv, dx)

                            vel[nec_n, t] = np.maximum(v + a * env.dt, 0.)  # action (n_car_show, ) filter out negative speed
                            pos[nec_n, t] = pos[nec_n, t - 1] + (v + vel[nec_n, t]) / 2 * env.dt  # new position
                            acc[nec_n, t] = 0. if v <= 0 else a

                            if pos[nec_n, t] >= env.max_p:
                                break

                        # draw in plot
                        # if nec_n > 0:
                        for i in fills:
                            mvdv = np.abs(acc[nec_n])
                            pos_n = pos[nec_n].copy()
                            pos_n[np.isnan(pos[nec_n])] = 0.
                            mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)] = np.minimum(
                                mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)], 1)

                            # fill y
                            bias = 2.3 * mvdv * i / (2 / 3 * n_colors)
                            top = pos[nec_n] - bias
                            down = pos[nec_n] + bias
                            plt.fill_between(x=x, y1=top, y2=down, color=cm.rainbow(g[i]))
                        nec_n += 1
                    else:
                        scatters.append(plt.scatter(init_t, 0., c='k'))
                        trajectory_x.append(init_t)
                        trajectory_y.append(0.)
                        print('\n------------\ndraw line\n-----------')
                else:
                    np.save('./results/pos%.1f' % av_rate, pos)
                    np.save('./results/vel%.1f' % av_rate, vel)
                    np.save('./results/acc%.1f' % av_rate, acc)

            fig.canvas.draw()

        # light time
        red_t, yellow_t = [], []
        for t in range(MAX_EP_STEP):
            t_in_cycle = t % (gd + yd + rd)
            if t_in_cycle < gd:
                is_red_light = False
            else:
                is_red_light = True
            if is_red_light and (t_in_cycle >= yd):
                red_t.append(LIGHT_P)
            else:
                red_t.append(np.nan)
            if is_red_light and (t_in_cycle < yd):
                yellow_t.append(LIGHT_P)
            else:
                yellow_t.append(np.nan)

        plt.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        plt.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)

    else:
        env = Env(light_p=LIGHT_P, ave_h=3600 / ave_flow, fix_start=True)
        par = './results/'
        labels = ['0% CAVs', '10% CAVs', '50% CAVs']
        colors = ['b', 'r', 'g']
        # mv test
        pos, vel, acc = [], [] ,[]
        for i in [0, 1, 5]:
            pos.append(np.load("%spos0.%i.npy" % (par, i)))
            vel.append(np.load("%svel0.%i.npy" % (par, i)))
            acc.append(np.load("%sacc0.%i.npy" % (par, i)))

        # light time
        gd, yd, rd = env.light_duration['green'] * 10, env.light_duration['yellow'] * 10, env.light_duration['red'] * 10
        red_t, yellow_t = [], []
        for t in range(MAX_EP_STEP):
            t_in_cycle = t % (gd + yd + rd)
            if t_in_cycle < gd:
                is_red_light = False
            else:
                is_red_light = True
            if is_red_light and (t_in_cycle >= yd):
                red_t.append(LIGHT_P)
            else:
                red_t.append(np.nan)
            if is_red_light and (t_in_cycle < yd):
                yellow_t.append(LIGHT_P)
            else:
                yellow_t.append(np.nan)

        e = []
        ittc = []
        travel_t = []
        for i in range(2):
            # emission for each car
            e.append(emission_car(acc[i][:nec], vel[i][:nec], Kij))
            # safety for each car
            ittc.append(iTTC(vel[i][:nec], pos[i][:nec], env.car_l))
            # travel time for each car
            travel_t.append(travel_time_car(pos[i][:nec]))

        n_colors = 15
        g = np.linspace(0, 1, n_colors)[::-1]
        x = np.linspace(0, MAX_EP_STEP, MAX_EP_STEP)
        fills = list(range(n_colors))[::-1]
        av_idx = [
            [-1],
            [3,12,20],
        ]

        fig = plt.figure(1, figsize=(10, 7))
        for i in range(2):
            ax1 = plt.subplot2grid((2, 6), (0, i*3), colspan=3)  # trajectory mv
            axins = zoomed_inset_axes(ax1, 4.5, loc=4)
            if i == 0:
                axins.axis([900, 900+350, 950, 950+200])
            else:
                axins.axis([200, 200+350, 10, 10+200])
            axins.set_yticks(())
            axins.set_xticks(())

            ax1.set_title('(%s) %s' % (chr(97+i), labels[i]))
            nec_n = 0
            for mp, ma in zip(pos[i][:nec], acc[i][:nec]):
                if nec_n in av_idx[i]:
                    ax1.plot(mp, c='b', alpha=1, lw=TRAJ_LW*2, solid_capstyle="butt")
                    axins.plot(mp, c='b', alpha=1, lw=TRAJ_LW*2, solid_capstyle="butt")
                else:
                    for j in fills:
                        mvdv = np.minimum(np.abs(ma), 4.)
                        pos_n = mp.copy()
                        pos_n[np.isnan(mp)] = 0.
                        mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)] = np.minimum(
                            mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)], 1)

                        # fill y
                        bias = 2.3 * mvdv * j / (2 / 3 * n_colors)
                        top = mp - bias
                        down = mp + bias
                        ax1.fill_between(x=x, y1=top, y2=down, color=cm.rainbow(g[j]), alpha=0.7)
                        axins.fill_between(x=x, y1=top, y2=down, color=cm.rainbow(g[j]), alpha=0.7)
                nec_n += 1

            ax1.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
            ax1.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
            axins.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
            axins.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
            ax1.set_xticklabels((np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
            ax1.set_xlim(xmin=0, xmax=MAX_EP_STEP)
            ax1.set_ylim(ymin=0)
            if i > 0:
                ax1.set_yticks(())
            else:
                ax1.set_ylabel('Position ($m$)')
            ax1.set_xlabel('Time ($s$)')

            if i == 0:
                mark_inset(ax1, axins, loc1=3, loc2=1, fc="none", ec="0.0")
            else:
                mark_inset(ax1, axins, loc1=4, loc2=2, fc="none", ec="0.0")

        ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)  # travel time
        for i in range(2):
            plt.scatter(np.arange(nec), travel_t[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
        plt.legend(loc='best', prop={'size': 6})
        plt.xlim((0, nec))
        plt.ylabel('Second')
        plt.xlabel('$n^{th}$ vehicle')
        plt.title('(c) Travel time')

        ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)  # emission
        for i in range(2):
            plt.scatter(np.arange(nec), e[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
        plt.legend(loc='best', prop={'size': 6})
        plt.xlim((0, nec))
        plt.ylabel('Liter')
        plt.xlabel('$n^{th}$ vehicle')
        plt.title('(d) Emission (VT-Micro model)')


        ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)  # iTTC
        for i in range(2):
            plt.scatter(np.arange(1, nec), ittc[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
        plt.legend(loc='best', prop={'size': 6})
        plt.ylabel('Crash risk')
        plt.xlim((0, nec))
        plt.xlabel('$n^{th}$ vehicle')
        plt.title('(e) Safety')

        plt.tight_layout(pad=0.2)
        if SAVE_FIG:
            plt.savefig('./results/demo3.png', format='png', dpi=500)
            plt.gcf().close()
        else:
            plt.show()


def plot_xiaobo_draw2(ave_flow, nec, av_rate):
    from matplotlib import cm
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    env = Env(light_p=LIGHT_P, ave_h=3600 / ave_flow, fix_start=True)
    par = './results/'
    labels = ['Trajectories by a classical car following model', 'Trajectories by a RNN based model']
    colors = ['b', 'r']
    # mv test
    pos, vel, acc = [], [] ,[]
    for i in [0, 0]:
        pos.append(np.load("%spos0.%i.npy" % (par, i)))
        vel.append(np.load("%svel0.%i.npy" % (par, i)))
        acc.append(np.load("%sacc0.%i.npy" % (par, i)))

    # light time
    gd, yd, rd = env.light_duration['green'] * 10, env.light_duration['yellow'] * 10, env.light_duration['red'] * 10
    red_t, yellow_t = [], []
    for t in range(MAX_EP_STEP):
        t_in_cycle = t % (gd + yd + rd)
        if t_in_cycle < gd:
            is_red_light = False
        else:
            is_red_light = True
        if is_red_light and (t_in_cycle >= yd):
            red_t.append(LIGHT_P)
        else:
            red_t.append(np.nan)
        if is_red_light and (t_in_cycle < yd):
            yellow_t.append(LIGHT_P)
        else:
            yellow_t.append(np.nan)

    e = []
    ittc = []
    travel_t = []
    for i in range(2):
        # emission for each car
        e.append(emission_car(acc[i][:nec], vel[i][:nec], Kij))
        # safety for each car
        ittc.append(iTTC(vel[i][:nec], pos[i][:nec], env.car_l))
        # travel time for each car
        travel_t.append(travel_time_car(pos[i][:nec]))

    n_colors = 15
    g = np.linspace(0, 1, n_colors)[::-1]
    x = np.linspace(0, MAX_EP_STEP, MAX_EP_STEP)
    fills = list(range(n_colors))[::-1]
    av_idx = [
        [-1],
        [3,12,20],
    ]

    fig = plt.figure(1, figsize=(10, 7))
    for i in range(2):
        ax1 = plt.subplot2grid((2, 6), (0, i*3), colspan=3)  # trajectory mv

        ax1.set_title('(%s) %s' % (chr(97+i), labels[i]))
        nec_n = 0
        for mp, ma in zip(pos[i][:nec], acc[i][:nec]):
            if nec_n in av_idx[0]:
                ax1.plot(mp, c='b', alpha=1, lw=TRAJ_LW*2, solid_capstyle="butt")
            else:
                if i == 0:
                    ax1.plot(mp, c='r', alpha=1, lw=TRAJ_LW * 2, solid_capstyle="butt")
                else:
                    for j in fills:
                        mvdv = np.minimum(np.abs(ma), 4.)
                        pos_n = mp.copy()
                        pos_n[np.isnan(mp)] = 0.
                        mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)] = np.minimum(
                            mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)], 1)

                        # fill y
                        bias = 2.3 * mvdv * j / (2 / 3 * n_colors)
                        top = mp - bias
                        down = mp + bias
                        ax1.fill_between(x=x, y1=top, y2=down, color=cm.rainbow(g[j]), alpha=0.7)
            nec_n += 1

        ax1.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        ax1.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        ax1.set_xticklabels((np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
        ax1.set_xlim((900, 900 + 350))
        ax1.set_ylim((950, 950 + 200))
        if i > 0:
            ax1.set_yticks(())
        else:
            ax1.set_ylabel('Position ($m$)')
        ax1.set_xlabel('Time ($s$)')

    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)  # travel time
    for i in range(2):
        plt.scatter(np.arange(nec), travel_t[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
    plt.legend(loc='best', prop={'size': 6})
    plt.xlim((0, nec))
    plt.ylabel('Second')
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(c) Travel time')

    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)  # emission
    for i in range(2):
        plt.scatter(np.arange(nec), e[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
    plt.legend(loc='best', prop={'size': 6})
    plt.xlim((0, nec))
    plt.ylabel('Liter')
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(d) Emission (VT-Micro model)')


    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)  # iTTC
    for i in range(2):
        plt.scatter(np.arange(1, nec), ittc[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
    plt.legend(loc='best', prop={'size': 6})
    plt.ylabel('Crash risk')
    plt.xlim((0, nec))
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(e) Safety')

    plt.tight_layout(pad=0.2)
    if SAVE_FIG:
        plt.savefig('./results/demo2.png', format='png', dpi=500)
        plt.gcf().close()
    else:
        plt.show()


def plot_xiaobo_draw3(ave_flow, nec, ave_h):
    from matplotlib import cm
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    env = Env(light_p=LIGHT_P, ave_h=3600 / ave_flow, fix_start=True)
    par = './results/'
    labels = ['Traditional vehicles', 'CAVs',]
    colors = ['r', 'b', ]
    # mv test
    pos, vel, acc = [], [] ,[]
    for i in [0]:
        pos.append(np.load("%spos0.%i.npy" % (par, i)))
        vel.append(np.load("%svel0.%i.npy" % (par, i)))
        acc.append(np.load("%sacc0.%i.npy" % (par, i)))

    env = Env(light_p=LIGHT_P, ave_h=ave_h, fix_start=True)
    env.set_fps(1000)
    rl = DDPG(
        s_dim=STATE_DIM, a_dim=ACTION_DIM, a_bound=ACTION_BOUND,
        train={'train': False, 'load_point': -1}, output_graph=False, model_dir=MODEL_DIR)
    rl.reset()

    # av test
    av_pos, av_vel, av_acc, red_t, yellow_t = av_loop(rl, env, return_light=True)
    pos.append(av_pos)
    vel.append(av_vel)
    acc.append(av_acc)

    # light time
    gd, yd, rd = env.light_duration['green'] * 10, env.light_duration['yellow'] * 10, env.light_duration['red'] * 10
    red_t, yellow_t = [], []
    for t in range(MAX_EP_STEP):
        t_in_cycle = t % (gd + yd + rd)
        if t_in_cycle < gd:
            is_red_light = False
        else:
            is_red_light = True
        if is_red_light and (t_in_cycle >= yd):
            red_t.append(LIGHT_P)
        else:
            red_t.append(np.nan)
        if is_red_light and (t_in_cycle < yd):
            yellow_t.append(LIGHT_P)
        else:
            yellow_t.append(np.nan)

    e = []
    ittc = []
    travel_t = []
    for i in range(2):
        # emission for each car
        e.append(emission_car(acc[i][:nec], vel[i][:nec], Kij))
        # safety for each car
        ittc.append(iTTC(vel[i][:nec], pos[i][:nec], env.car_l))
        # travel time for each car
        travel_t.append(travel_time_car(pos[i][:nec]))

    n_colors = 15
    g = np.linspace(0, 1, n_colors)[::-1]
    x = np.linspace(0, MAX_EP_STEP, MAX_EP_STEP)
    fills = list(range(n_colors))[::-1]

    fig = plt.figure(1, figsize=(10, 7))
    for i in range(2):
        ax1 = plt.subplot2grid((2, 6), (0, i*3), colspan=3)  # trajectory mv

        if i == 0:
            axins = zoomed_inset_axes(ax1, 4.5, loc=4)
            axins.axis([900, 900+350, 950, 950+200])
            axins.set_yticks(())
            axins.set_xticks(())

        ax1.set_title('(%s) %s' % (chr(97+i), labels[i]))
        nec_n = 0
        for mp, ma in zip(pos[i][:nec], acc[i][:nec]):
            if i == 0:
                for j in fills:
                    mvdv = np.minimum(np.abs(ma), 4.)
                    pos_n = mp.copy()
                    pos_n[np.isnan(mp)] = 0.
                    mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)] = np.minimum(
                        mvdv[(LIGHT_P - 7 < pos_n) & (pos_n < LIGHT_P + 4)], 1)

                    # fill y
                    bias = 2.3 * mvdv * j / (2 / 3 * n_colors)
                    top = mp - bias
                    down = mp + bias
                    ax1.fill_between(x=x, y1=top, y2=down, color=cm.rainbow(g[j]), alpha=0.7)
                    axins.fill_between(x=x, y1=top, y2=down, color=cm.rainbow(g[j]), alpha=0.7)
            else:
                ax1.plot(mp, c='b', alpha=0.9, lw=TRAJ_LW * 2, solid_capstyle="butt")
            nec_n += 1

        ax1.plot(np.arange(len(red_t)), red_t, 'r', lw=LIGHT_LW, solid_capstyle="butt")
        ax1.plot(np.arange(len(yellow_t)), yellow_t, c='y', lw=LIGHT_LW, solid_capstyle="butt")
        ax1.set_xticklabels((np.arange(0, MAX_EP_STEP, 500) * env.dt).astype(np.int32))
        ax1.set_xlim(xmin=0, xmax=MAX_EP_STEP)
        ax1.set_ylim(ymin=0)
        if i > 0:
            ax1.set_yticks(())
        else:
            ax1.set_ylabel('Position ($m$)')
        ax1.set_xlabel('Time ($s$)')

        if i == 0:
            mark_inset(ax1, axins, loc1=3, loc2=1, fc="none", ec="0.0")

    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)  # travel time
    for i in range(2):
        plt.scatter(np.arange(nec), travel_t[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
    plt.legend(loc='best', prop={'size': 6})
    plt.xlim((0, nec))
    plt.ylabel('Second')
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(c) Travel time')
    plt.text(5 - 2, 80, 'Red phase 1', color='r', fontsize=8)
    plt.text(11 - 2, 120, 'Red phase 2', color='r', fontsize=8)
    plt.text(17 - 2, 160, 'Red phase 3', color='r', fontsize=8)
    plt.text(23 - 3, 195, 'Red phase 4', color='r', fontsize=8)
    plt.text(14 - 4, 50, 'Red phase', color='b', fontsize=8)

    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)  # emission
    for i in range(2):
        plt.scatter(np.arange(nec), e[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
    plt.legend(loc='best', prop={'size': 6})
    plt.xlim((0, nec))
    plt.ylabel('Liter')
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(d) Emission (VT-Micro model)')
    plt.text(5 - 2, 0.055, 'Red phase 1', color='r', fontsize=8)
    plt.text(11 - 2, 0.07, 'Red phase 2', color='r', fontsize=8)
    plt.text(17 - 2, 0.088, 'Red phase 3', color='r', fontsize=8)
    plt.text(23 - 3, 0.105, 'Red phase 4', color='r', fontsize=8)
    plt.text(14 - 4, 0.04, 'Red phase', color='b', fontsize=8)

    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)  # iTTC
    for i in range(2):
        plt.scatter(np.arange(1, nec), ittc[i], s=20, c=colors[i], alpha=0.5, label=labels[i])
    plt.legend(loc='best', prop={'size': 6})
    plt.ylabel('Crash risk')
    plt.xlim((0, nec))
    plt.xlabel('$n^{th}$ vehicle')
    plt.title('(e) Safety')
    plt.text(5 - 2, 45, 'Red phase 1', color='r', fontsize=8)
    plt.text(11 - 2, 60, 'Red phase 2', color='r', fontsize=8)
    plt.text(17 - 2, 70, 'Red phase 3', color='r', fontsize=8)
    plt.text(23 - 3, 85, 'Red phase 4', color='r', fontsize=8)
    plt.text(14 - 2, 10, 'Red phase', color='b', fontsize=8)

    plt.tight_layout(pad=0.2)
    if SAVE_FIG:
        plt.savefig('./results/demo3.png', format='png', dpi=500)
        plt.gcf().close()
    else:
        plt.show()


def choose_plot(plot_n):
    config = [
        dict(fun=plot_av_mv,  # 0
             kwargs={"ave_h": 3.}),  # 1500 veh/h
        dict(fun=plot_av_diff_h,  # 1
             kwargs={'h_list': [5., 4., 3.], }),
        dict(fun=av_diff_light_duration,  # 2
             kwargs={'l_duration': [45, 35, 25]}),
        dict(fun=plot_reward,  # 3
             kwargs={'download': False, 'n_models': 6}),
        dict(fun=plot_mix_traffic,  # 4
             kwargs={'av_rate': np.arange(0, 1.1, 0.5)}),
        dict(fun=plot_demand_change,  # 5
             kwargs={'demand_change': [900, 1200, 1500]}),
        dict(fun=plot_xiaobo_demo1,  # 6
             kwargs={'ave_flow': 1300, 'nec': [0, 29]}),
        dict(fun=plot_xiaobo_demo2,  # 7
             kwargs={'ave_flow': 1300, 'nec': 29, 'av_rate': 0.1, 'drawing': False, 'save': False},
             ),
        dict(fun=plot_xiaobo_draw2,  # 8
             kwargs={'ave_flow': 1300, 'nec': 29, 'av_rate': 0}
             ),
        dict(fun=plot_xiaobo_draw3,  # 9
             kwargs={'ave_flow': 1300, 'nec': 29, 'ave_h': 2.7692}
             ),
    ][plot_n]
    config['fun'](**config['kwargs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_n', type=int, nargs='+', default=[0], choices=range(6),
                        help='The model number')
    parser.add_argument('-p', '--plot_n', type=int, nargs='+', default=[1, 2], choices=range(10),
                        help='The plot function number')
    parser.add_argument('-o', '--output', action='store_true')

    args = parser.parse_args(
        # args='-m 0 -p 1'.split()
    )
    for model_n in args.model_n:
        MODEL_DIR = './tf_models/%s' % model_n
        SAVE_FIG = args.output
        for plot_n in args.plot_n:
            t0 = time.time()
            choose_plot(plot_n)
            print(time.time()-t0)