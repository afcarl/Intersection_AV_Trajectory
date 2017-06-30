import numpy as np
import pyglet
from matplotlib import cm
import matplotlib.pyplot as plt


pyglet.clock.set_fps_limit(1000)


def convert2pixel(meter):
    scale = 10
    pixel = meter * scale
    return pixel


class Env(object):
    action_bound = [-2, 2]
    action_dim = 1
    state_dim = 6
    dt = 0.1    # driving refresh rate
    ave_headway = 2    # used for the interval of generate car
    light_duration = {'red': 20., 'green': 20.}     # seconds
    car_l = 4.
    init_v = 60.    # km/h
    car_num_limit = 150
    max_p = 700
    light_p = max_p * .95
    viewer = None

    def __init__(self):
        # position(m), velocity(m/s), passed_light, run_red, reward
        self.car_info = np.zeros((self.car_num_limit, 5), dtype=np.float32)
        self.t_gen = 0.  # counting time for generate car
        self.t_light = 0.  # counting time for light change
        self.is_red_light = True  # red phase
        self.n_car_show = 0

    def step(self, action):
        cars_show = self.car_info[:self.n_car_show]
        pass_light_last_step = cars_show[:, 2].copy()
        v_ = cars_show[:, 1] + action * self.dt  # action (n_car_show, )
        v_ = v_.clip(min=0.)    # stop
        cars_show[:, 0] += (cars_show[:, 1] + v_) / 2 * self.dt  # new position
        cars_show[:, 1] = v_
        cars_show[:, 2] = cars_show[:, 0] > self.light_p   # new state of passed_light

        self._check_change_light()  # traffic light changes
        t2r, t2g = self._get_t2r_and_t2g()  # time to red and green light
        s = self._get_state(t2r, t2g, cars_show)  # [t2r_norm, t2g_norm, dist2light_norm, dx_norm, dv, v_norm]
        r, done = self._get_r_and_done(s, pass_light_last_step, cars_show, t2r, t2g)
        cars_show[:, -1] = r     # record reward for plotting on screen
        return s, r, done,

    def update_s_(self, s_):
        # check and changes state
        s_ = self._check_gen_car(s_)  # check to generate car
        s_ = self._check_rm_car(s_)  # check to remove who achieve the max_p
        return s_

    def reset(self):
        self.t_gen = 0.  # counting time for generate car
        c = np.random.choice(['red', 'green'])
        self.t_light = np.random.uniform(0, self.light_duration[c])  # counting time for light change
        self.is_red_light = True if c == 'red' else False   # red phase
        if self.viewer is not None:
            self.viewer.light.colors = self.viewer.color[c] * 4
        self.n_car_show = 0
        self.car_info *= 0
        self._check_gen_car()
        s = self._get_state(*self._get_t2r_and_t2g(), self.car_info[:self.n_car_show])
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.car_num_limit, self.max_p, self.light_p)

        for i, car in enumerate(self.viewer.cars):
            if i < self.n_car_show:
                car.visible = True
                car.update(self.car_info[i])
            else:
                car.visible = False
                car.label.text = ''
        self.viewer.render()

    def sample_action(self):    # action (n_car_show, )
        # return np.zeros((self.n_car_show, ))
        return np.random.uniform(*self.action_bound, self.n_car_show)

    def set_fps(self, fps=10):
        pyglet.clock.set_fps_limit(fps)

    def _check_change_light(self):
        self.t_light += self.dt
        if self.is_red_light:
            if self.t_light >= self.light_duration['red']:  # turn green
                self.t_light -= self.light_duration['red']
                self.is_red_light = False
                if self.viewer is not None:
                    self.viewer.light.colors = self.viewer.color['green'] * 4
        else:
            if self.t_light >= self.light_duration['green']:    # turn red
                self.t_light -= self.light_duration['green']
                self.is_red_light = True
                if self.viewer is not None:
                    self.viewer.light.colors = self.viewer.color['red'] * 4

    def _check_gen_car(self, s_=None):
        if self.t_gen >= self.ave_headway or len(self.car_info.nonzero()[0]) == 0:
            self.t_gen -= self.ave_headway
            self.car_info[self.n_car_show, :] = [0, self.init_v/3.6, False, False, 0]   # position, velocity, passed_light, run_red, reward
            self.n_car_show += 1
            if s_ is not None:
                cars_show = self.car_info[:self.n_car_show]
                s_ = self._get_state(*self._get_t2r_and_t2g(), cars_show)
        self.t_gen += self.dt   # accumulate time
        return s_

    def _check_rm_car(self, s_):
        if self.car_info[0, 0] >= self.max_p:
            self.car_info[0, :] *= 0
            self.car_info = np.roll(self.car_info, -1, axis=0)      # move the 1st row to last, keep its id in memory
            self.n_car_show -= 1    # minus one car on the road
            if s_ is not None:
                cars_show = self.car_info[:self.n_car_show]
                s_ = self._get_state(*self._get_t2r_and_t2g(), cars_show)
        return s_

    def _get_state(self, t2r, t2g, cars):
        """
        0. t2red / 100     (normalized)
        1. t2green / 100      (normalized)
        2. distance2light / 100                (normalized)
        3. dx     (exclude car length, normalized)
        4. dv     (m/s)   (normalized)
        5. v      (m/s)   (normalized)
        # 6. time2light
        """
        p, v = cars[:, 0], cars[:, 1]
        t2r_vec = (np.zeros_like(p) + t2r) / 100.     # normalized
        t2g_vec = (np.zeros_like(p) + t2g) / 100.   # normalized
        dp = self.light_p - p
        distance2light = np.minimum(dp, 1000.) / 1000.     # normalized affect range=1000, normalized to 10
        distance2light[dp < 0] = 1.     # if passed light, the distance set to 10

        dx_max = 100.    # meters if no preceding car
        dx = np.concatenate(([dx_max], -np.diff(p)))     # add dx_max to the leader
        dx = (dx.clip(max=dx_max) - self.car_l) / dx_max    # clip to the max dx and normalize
        dv = np.concatenate(([0.], np.diff(v))) / 5         # add dv=0 as for leader
        v_norm = v / (self.init_v/3.6)    # normalize velocity
        return np.vstack((t2r_vec, t2g_vec, distance2light, dx, dv, v_norm)).T

    def _get_state2(self, t2r, t2g, cars):
        p, v = cars[:, 0], cars[:, 1]
        # t2r_vec = (np.zeros_like(p) + t2r)
        # t2g_vec = (np.zeros_like(p) + t2g)
        dp = self.light_p - p
        distance2light = np.minimum(dp, 1000.) / 1000.  # normalized affect range=1000
        distance2light[dp < 0] = 1.  # if passed light, the distance set to 1

        dx_max = 100.  # meters if no preceding car
        dx = np.concatenate(([dx_max], -np.diff(p)))  # add dx_max to the leader
        dx = (dx.clip(max=dx_max) - self.car_l) / dx_max  # clip to the max dx and normalize
        dv = np.concatenate(([0.], np.diff(v))) / 5  # add dv=0 as for leader
        v_norm = v / self.init_v  # normalize velocity
        t2light = np.empty_like(dp)
        t2light[dp>=0] = np.minimum((dp / (v + 1e-3))[dp>=0], 100.) # not pass light
        # t2light[dp<0] = (np.zeros_like(dp[dp < 0]) + 100)    # passed light
        run_green = np.ones_like(p)
        t_red = self.light_duration['red']
        t_green = self.light_duration['green']
        if self.is_red_light:
            pass_index = (t2light > t2g) & (t2light < (t2g + t_green))
            stop_index = t2light < t2g
            run_green[pass_index] = (t2light[pass_index] - t2g)/t_green   # run to next green phase
            run_green[stop_index] = -(t2g-t2light[stop_index])/t_red      # run to this red phase
            # others not cares, = 1
        else:
            pass_index = (t2light > t2g) & (t2light < (t2g + t_green))
            stop_index = t2light < t2g
            run_green[t2light < t2r] = 1.   # run to this green phase
            # run_green[t2light > (t2r + t_red)] = 0. # others not cares,
            # others run to next red phase = -1
        run_green[dp<0] = 1.      # passed light = run_green
        # t2g_t2l = (t2g - t2light)/20
        return np.vstack((run_green, dx, dv, v_norm)).T

    def _get_t2r_and_t2g(self):    # time to next red and green phase
        if self.is_red_light:  # time 2 next red and green
            t2g = self.light_duration['red'] - self.t_light  # rest of red time
            t2r = t2g + self.light_duration['green']  # rest of red time + a green phase
        else:  # time 2 next red and green
            t2r = self.light_duration['green'] - self.t_light  # rest of green time
            t2g = t2r + self.light_duration['red']  # rest of green time + a red phase
        return t2r, t2g

    def _get_r_and_done(self, state, pass_light_last_step, cars_show, t2r, t2g):
        """
        collision: r -= 5
        run the red light: r -= 10
        maintain high speed: r += max(1)
        """
        done = False
        # r = np.zeros((self.n_car_show, ))
        v_r_max = 1.
        r = v_r_max - np.abs(cars_show[:, 1] * 3.6 - self.init_v)/(self.init_v/(2*v_r_max))      # normalized (v=30km/h -> r += 1, v=0km/h or =60 -> r -= 1

        # check if run red
        # if self.is_red_light:   # red phase
        #     run_red = (np.invert(pass_light_last_step.astype(np.bool))) & cars_show[:, 2].astype(np.bool)
            # r[run_red] = -20.  # run red
            # if np.any(run_red):
                # done = True
        # else:   # green phase
        #     run_green = (np.invert(pass_light_last_step.astype(np.bool))) & cars_show[:, 2].astype(np.bool)
        #     r[run_green] = 1.  # run green

        dp = self.light_p - cars_show[:, 0]
        t2light = np.zeros_like(dp) + 100.  # default to max time
        t2light[dp >= 0] = np.minimum((dp / (cars_show[:, 1] + 1e-3))[dp >= 0], 100.)  # not pass light
        t_green = self.light_duration['green']
        t_red = self.light_duration['red']
        if self.is_red_light:
            # not_care_index = t2light >= (t2g + t_green)
            # pass_index = (t2light > t2g) & (t2light < (t2g + t_green))
            not_pass_index = (t2light <= t2g) #| ((t2light > t2r) & (t2light < t2r + t_red))

            # dist2light = state[:, 2]*1000   # scale back to meter
            # short_dist2light_idx = dist2light < 3
            # r[short_dist2light_idx] += -5.      # to close to light
        else:
            # pass_index = t2light < t2r
            not_pass_index = (t2light >= t2r) & (t2light < t2g)
        # r[pass_index | (dp <= 0)] += .1   # run to this green phase
        r[not_pass_index] = -v_r_max  # set to the min of reward for v to avoid stopping effect

        # check collision and too close distance
        is_collision = state[:, 3] * 100 <= 0     # dx
        headway = state[:, 3] * 100 / (cars_show[:, 1]*3.6+1e-4)
        too_close1 = headway <= 0.2  #  dx/v = headway normalized
        r[too_close1] = -v_r_max
        # too_close2 = headway <= 0.1  # dx/v = headway normalized
        # r[too_close2] = -v_r_max
        if np.any(is_collision): #or run_red_terminal:    # dx < 0 = collision
            done = True
            # r[is_collision] = -5.
        return r, done

    def plot_reward_func(self):
        pass


class Viewer(pyglet.window.Window):
    color = {
        'background': [100/255]*3 + [1],
        'red': [249, 67, 42],
        'green': [74, 214, 49]
    }
    fps_display = pyglet.clock.ClockDisplay()
    car_img = pyglet.image.load('car.png')

    def __init__(self, car_num_limit, max_p, light_p, width=600, height=600,):
        super(Viewer, self).__init__(width, height, resizable=False, caption='RL_car', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        self.set_icon(pyglet.image.load('car.png'))
        pyglet.gl.glClearColor(*self.color['background'])

        min_length = min([width, height])
        center_coord = np.array([min_length, min_length])/2
        self.batch = pyglet.graphics.Batch()
        self.cars = [Car(self.car_img, self.batch, center_coord, max_p) for _ in range(car_num_limit)]

        cr = center_coord[0] * 0.9
        mod_p = convert2pixel(light_p) % (np.pi * np.square(cr))
        radian = mod_p / cr
        shrink_cr = cr * light_p / max_p  # shrink the central radius, dx, dy are shorter
        dx = np.sin(radian) * shrink_cr
        dy = np.cos(radian) * shrink_cr
        light_coord = center_coord[0] + dx, center_coord[1] - dy
        light_box = [light_coord[0] - 10, light_coord[1] - 10,
                     light_coord[0] + 10, light_coord[1] - 10,
                     light_coord[0] + 10, light_coord[1] + 10,
                     light_coord[0] - 10, light_coord[1] + 10]
        self.light = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', light_box), ('c3B', self.color['red']*4))

    def render(self):
        pyglet.clock.tick()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()

    def on_key_press(self, symbol, modifiers):
        if modifiers & pyglet.window.key.MOD_CTRL and symbol == pyglet.window.key.L:
            pass
        elif symbol == pyglet.window.key.UP:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.DOWN:
            pyglet.clock.set_fps_limit(10)


class Car(pyglet.sprite.Sprite):
    l = 4  # length of car

    def __init__(self, img, batch, geo_center, max_p):
        super(Car, self).__init__(img=img, batch=batch, usage='stream', )

        # new car attributes
        self.max_p = max_p
        self.gc, self.cr = geo_center, geo_center[0] * 0.9  # geo center and center radius, unit=pixel
        self.g_circumference = np.pi * np.square(self.cr)   # unit=pixel

        # text describe
        self.label = pyglet.text.Label(
            text='', font_size=12, bold=True,
            anchor_x='center', anchor_y='center', align='center', batch=batch)

        # supered attributes
        self.visible = False
        self.scale = convert2pixel(self.l) / self.image.width    # width is the car length
        self.image.anchor_x = self.image.width / 2
        self.image.anchor_y = self.image.height / 2

    def update(self, car_info):
        p, v, r = car_info[0], car_info[1]*3.6, car_info[-1]

        p_percent = p / self.max_p  # how many road left
        mod_p = convert2pixel(p) % self.g_circumference
        radian = mod_p / self.cr
        shrink_cr = self.cr * p_percent    # shrink the central radius, dx, dy are shorter
        dx = np.sin(radian) * shrink_cr
        dy = np.cos(radian) * shrink_cr

        # these are supered attributes
        self.scale = (p_percent+.1) * convert2pixel(self.l) / self.image.width  # shrink the image size
        self.x, self.y, self.rotation = self.gc[0] + dx, self.gc[1] - dy, -np.rad2deg(radian)

        # label
        r_color = cm.rainbow(int((-r+0.4)*310-20))  # the value should in a range of (0, 255), the RGB max=1
        self.label.text = '%.0f/%.2f' % (v, r)
        self.label.x, self.label.y = int(self.gc[0] + dx*1.2), int(self.gc[1] - dy*1.2)
        self.label.color = (np.array(r_color)*255).astype(np.int32).tolist()    # has to be a tuple or list with max=255


if __name__ == '__main__':
    np.random.seed(1)
    env = Env()
    # env.plot_reward_func()
    env.set_fps(60)

    for i in range(100):
        s = env.reset()
        while True:
            env.render()
            a = np.zeros((env.n_car_show, ))
            s_, r, done = env.step(a)
            s = env.update_s_(s_)
            if done:
                break
