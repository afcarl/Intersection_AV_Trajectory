import tensorflow as tf
import numpy as np
import os
import shutil
from numba import njit
from memory import Memory

@njit
def sample(data, m_capacity, batch_size):
    indices = np.random.randint(0, m_capacity, size=batch_size)
    return data[indices]


class DDPG(object):
    ep_r = 0.   # recode it in training

    def __init__(
            self,
            s_dim, a_dim, a_bound, a_lr=0.001, c_lr=0.001, tau=0.001, gamma=0.9,
            memory_capacity=5000, batch_size=64,
            train={'train': True, 'save_iter': None, 'load_point': -1},
            model_dir='./tf_models/0', log_dir='./log', output_graph=True,
        ):
        tf.reset_default_graph()

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.output_graph = output_graph
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.train_ = train
        self.log_dir = log_dir
        self.learn_counter = 0
        self.save_model_freq = 1000 if train.get('save_iter') is None else train['save_iter']

        self.memory_capacity = memory_capacity
        self.memory = np.empty(
            self.memory_capacity, dtype=[('s', np.float32, (s_dim,)), ('a', np.float32, (a_dim,)),
                                         ('r', np.float32, (1,)), ('s_', np.float32, (s_dim,))])
        self.pointer = 0
        self.update_times = 1  # TODO: inner loop for update
        self.batch_holder = np.empty(
            self.batch_size*self.update_times, dtype=[('s', np.float32, (s_dim,)), ('a', np.float32, (a_dim,)),
                                                      ('r', np.float32, (1,)), ('s_', np.float32, (s_dim,))])

        # inputs
        self.S = tf.placeholder(tf.float32, shape=[None, s_dim], name='s')
        self.R = tf.placeholder(tf.float32, [None, 1], name='r')
        self.S_ = tf.placeholder(tf.float32, shape=[None, s_dim], name='s_')
        self.tfep_r = tf.placeholder(tf.float32, shape=None, name='ep_r')
        tf.summary.scalar('ep_reward', self.tfep_r)

        with tf.variable_scope('Actor'):
            self.a = self._build_a_net(self.S, scope='eval_net', trainable=True)
            a_ = self._build_a_net(self.S_, scope='target_net', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c_net(self.S, self.a, 'eval_net', trainable=True)
            q_ = self._build_c_net(self.S_, a_, 'target_net', trainable=False)
        with tf.variable_scope('update_target'):
            ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
            at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
            ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')
            self.update_target_op = [[tf.assign(at, tau * ae + (1.-tau) * at), tf.assign(ct, tau * ce + (1.-tau) * ct)]
                                     for at, ae, ct, ce in zip(at_params, ae_params, ct_params, ce_params)]
        with tf.variable_scope('target_q'):
            target_q = self.R + gamma * q_

        c_loss, self.ISWeights, self.abs_errors = self._class_depended_graph(target_q, q)      # self.ISWeights and abs_loss is None here

        with tf.variable_scope('C_train'):
            self.c_train_op = tf.train.RMSPropOptimizer(c_lr).minimize(c_loss, var_list=ce_params)
        with tf.variable_scope('a_loss'):
            policy_loss = -tf.reduce_mean(q)# - 0.01*tf.square(self.a))    # TODO: acceleration penalty
            tf.summary.scalar('actor_loss', policy_loss)
        with tf.variable_scope('A_train'):
            self.a_train_op = tf.train.RMSPropOptimizer(a_lr).minimize(policy_loss, var_list=ae_params)

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=100)
        if not self.train_['train']:
            if self.train_['load_point'] == -1:
                ckpt = tf.train.get_checkpoint_state(self.model_dir, 'checkpoint').all_model_checkpoint_paths[-1]
            else:
                ckpt = self.model_dir + '/DDPG.ckpt-%i' % self.train_['load_point']
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.global_variables_initializer())

        if output_graph:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir, ignore_errors=True)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def _class_depended_graph(self, target_q, q):
        with tf.variable_scope('c_loss'):
            c_loss = tf.reduce_mean(tf.squared_difference(target_q, q))
            tf.summary.scalar('critic_loss', c_loss)
        return c_loss, None, None

    def _build_a_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.truncated_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 256, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l1')
            tf.summary.histogram('layer1', net)
            net = tf.layers.dense(net, 256, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l2')
            tf.summary.histogram('layer2', net)
            net = tf.layers.dense(net, 256, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l3')
            tf.summary.histogram('layer3', net)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=tf.constant_initializer(-np.mean(self.a_bound)),
                                          name='a', trainable=trainable)
                scale = tf.convert_to_tensor((self.a_bound[1] - self.a_bound[0])/2, dtype=tf.float32, name='scale')
                shift = tf.convert_to_tensor(self.a_bound[1] - scale, dtype=tf.float32, name='shift')
                scaled_a = tf.multiply(actions, scale, name='scaled_a')
                shifted_a = tf.add(scaled_a, shift, name='shifted_a')
            tf.summary.histogram('action', shifted_a)
        return shifted_a

    def _build_c_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.truncated_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 256
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            tf.summary.histogram('layer1', net)

            net = tf.layers.dense(net, 256, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l2')
            tf.summary.histogram('layer2', net)
            net = tf.layers.dense(net, 256, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l3')
            tf.summary.histogram('layer3', net)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, trainable=trainable)   # Q(s,a)
            tf.summary.histogram('q_value', q)
        return q

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S: s})  # shape (n_cars, 1)

    def learn(self):   # batch update
        self.sess.run(self.update_target_op)  # soft update
        # indices = np.random.randint(self.memory_capacity, size=self.batch_size * self.update_times)
        self.batch_holder[:] = sample(self.memory, self.memory_capacity, int(self.batch_size*self.update_times))
        # np.take(self.memory, indices, axis=0, out=self.batch_holder)

        s, a, r, s_ = self.batch_holder['s'], self.batch_holder['a'], self.batch_holder['r'], self.batch_holder['s_']

        for ut in range(self.update_times):
            self.learn_counter += 1
            bs, ba, br, bs_ = s[ut * self.batch_size: (ut + 1) * self.batch_size], \
                              a[ut * self.batch_size: (ut + 1) * self.batch_size], \
                              r[ut * self.batch_size: (ut + 1) * self.batch_size], \
                              s_[ut * self.batch_size: (ut + 1) * self.batch_size]
            self.sess.run(self.c_train_op, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            self.sess.run(self.a_train_op, {self.S: bs, self.R: br, self.S_: bs_})
            self._record_learning(bs, ba, br, bs_)      # add summary and save ckpt

    def _record_learning(self, bs, ba, br, bs_, ISw=None):
        if self.output_graph and self.learn_counter % self.save_model_freq == 0:
            feed_dict = {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.tfep_r: self.ep_r}
            if ISw is not None: feed_dict[self.ISWeights] = ISw
            merged = self.sess.run(self.merged, feed_dict)
            self.writer.add_summary(merged, global_step=self.learn_counter)

        if self.train_['save_iter'] is not None:
            if self.learn_counter % self.train_['save_iter'] == 0:    # save model periodically
                ckpt_path = os.path.join(self.model_dir, 'DDPG.ckpt')
                save_path = self.saver.save(self.sess, ckpt_path, global_step=self.learn_counter, write_meta_graph=False)
                print("\nSave Model %s\n" % save_path)

    def threadlearn(self, stop_event):
        while not stop_event.is_set():
            self.learn()

    def store_transition(self, s, a, r, s_):
        for item in [s,a,r,s_]:
            assert item.dtype == np.float32
        if a.ndim < 2:
            a = a[:, None]
        if r.ndim < 2:
            r = r[:, None]
        # s = np.ascontiguousarray(s)
        # s_ = np.ascontiguousarray(s_)
        p_ = self.pointer + r.size
        if p_ <= self.memory_capacity:
            self.memory['s'][self.pointer:p_] = s
            self.memory['a'][self.pointer:p_] = a
            self.memory['r'][self.pointer:p_] = r
            self.memory['s_'][self.pointer:p_] = s_
        else:
            p_ = p_ % self.memory_capacity
            self.memory['s'][self.pointer:] = s[:r.size - p_]
            self.memory['a'][self.pointer:] = a[:r.size - p_]
            self.memory['r'][self.pointer:] = r[:r.size - p_]
            self.memory['s_'][self.pointer:] = s_[:r.size - p_]
            self.memory['s'][:p_] = s[-p_:]
            self.memory['a'][:p_] = a[-p_:]
            self.memory['r'][:p_] = r[-p_:]
            self.memory['s_'][:p_] = s_[-p_:]
        self.pointer = p_

    def save(self, path=None):
        path = path if path is not None else self.model_dir
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
        ckpt_path = os.path.join(path, 'DDPG.ckpt')
        save_path = self.saver.save(self.sess, ckpt_path, global_step=self.learn_counter, write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)


class DDPGPrioritizedReplay(DDPG):
    log_dir = './log'
    save_model_iter = 100000

    def __init__(
            self,
            s_dim, a_dim, a_bound, a_lr=0.001, c_lr=0.001, tau=0.001, gamma=0.9,
            memory_capacity=5000, batch_size=64,
            train={'train': True, 'save_iter': None, 'load_point': -1},
            model_dir='./tf_models', log_dir='./log', output_graph=True,
    ):
        super(DDPGPrioritizedReplay, self).__init__(
            s_dim=s_dim, a_dim=a_dim, a_bound=a_bound, a_lr=a_lr, c_lr=c_lr,
            tau=tau, gamma=gamma, memory_capacity=memory_capacity, batch_size=batch_size,
            train=train,
            model_dir=model_dir, log_dir=log_dir, output_graph=output_graph,
        )
        self.memory = Memory(self.memory_capacity, self.batch_size, self.s_dim, self.a_dim)

    def _class_depended_graph(self, target_q, q):
        ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('abs_errors'):
            abs_errors = tf.abs(target_q - q)  # for updating Sumtree
        with tf.variable_scope('c_loss'):
            c_loss = tf.reduce_mean(ISWeights * tf.squared_difference(target_q, q))
            tf.summary.scalar('critic_loss', c_loss)
        return c_loss, ISWeights, abs_errors

    def learn(self, lock=None):   # batch update
        self.sess.run(self.update_target_op)  # soft update target
        for _ in range(self.update_times):
            self.learn_counter += 1
            if lock is not None: lock.acquire()
            tree_idx, bt, ISWeights = self.memory.sample()
            abs_errors, _ = self.sess.run(
                [self.abs_errors, self.c_train_op],
                {self.S: bt['s'], self.a: bt['a'], self.R: bt['r'], self.S_: bt['s_'], self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)
            if lock is not None: lock.release()
            self.sess.run(self.a_train_op, {self.S: bt['s'], self.R: bt['r'], self.S_: bt['s_']})
            self._record_learning(bt['s'], bt['a'], bt['r'], bt['s_'], ISWeights)      # add summary and save ckpt

    def store_transition(self, s, a, r, s_):
        if a.ndim < 2:
            a = a[:, None]
        if r.ndim < 2:
            r = r[:, None]
        # s = np.ascontiguousarray(s)
        # s_ = np.ascontiguousarray(s_)
        self.memory.store(s, a, r, s_)
