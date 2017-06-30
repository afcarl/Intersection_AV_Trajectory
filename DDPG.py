import tensorflow as tf
import numpy as np
import os
import shutil


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        for d in data:
            leaf_idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = d  # update data_frame
            self.update(leaf_idx, p)  # update tree_frame

            self.data_pointer += 1
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound - self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)   # set the max p for new p

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [None] * n, [None] * n, [None] * n
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights[i] = self.tree.capacity * prob
            batch_idx[i] = idx
            batch_memory[i] = data

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)


class DDPG(object):
    log_dir = './log'
    save_model_iter = 100000

    def __init__(
            self,
            s_dim, a_dim, a_bound,
            a_lr=0.001, a_replace_iter=600,
            c_lr=0.001, c_replace_iter=500, gamma=0.9,
            memory_capacity=5000, batch_size=64,
            output_graph=False, restore=False,
            periodical_save=False, model_dir='./model',
                 ):
        tf.reset_default_graph()

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.a_replace_iter = a_replace_iter
        self.c_replace_iter = c_replace_iter
        self.a_replace_counter = 0
        self.c_replace_counter = 0
        self.output_graph = output_graph
        self.periodical_save = periodical_save
        self.model_dir = model_dir
        self.batch_size = batch_size

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, s_dim*2+a_dim+1), dtype=np.float32)
        self.pointer = 0

        # inputs
        self.S = tf.placeholder(tf.float32, shape=[None, s_dim], name='s')
        self.R = tf.placeholder(tf.float32, [None, 1], name='r')
        self.S_ = tf.placeholder(tf.float32, shape=[None, s_dim], name='s_')

        with tf.variable_scope('Actor'):
            self.a = self._build_a_net(self.S, scope='eval_net', trainable=True)
            a_ = self._build_a_net(self.S_, scope='target_net', trainable=False)

        with tf.variable_scope('Critic'):
            q = self._build_c_net(self.S, self.a, 'eval_net', trainable=True)
            q_ = self._build_c_net(self.S_, a_, 'target_net', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            target_q = self.R + gamma * q_

        with tf.variable_scope('c_loss'):
            c_loss = tf.reduce_mean(tf.squared_difference(target_q, q))
            tf.summary.scalar('critic_loss', c_loss)

        with tf.variable_scope('C_train'):
            self.c_train_op = tf.train.AdamOptimizer(c_lr).minimize(c_loss, global_step=self.global_step, var_list=self.ce_params)

        policy_loss = - tf.reduce_mean(q)
        self.a_train_op = tf.train.AdamOptimizer(a_lr).minimize(policy_loss, global_step=self.global_step, var_list=self.ae_params)

        self.sess = tf.Session()

        self.saver = tf.train.Saver(max_to_keep=100)
        if restore['load']:
            # all_ckpt = tf.train.get_checkpoint_state(self.model_dir, 'checkpoint').all_model_checkpoint_paths
            self.saver.restore(self.sess, restore['point'])
        else:   # clear path for saving model
            if os.path.isdir(self.model_dir):
                shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)
            self.sess.run(tf.global_variables_initializer())

        if output_graph:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def _build_a_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)
            net = tf.layers.dense(s, 50, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l1')
            tf.summary.histogram('layer1', net)
            net = tf.layers.dense(net, 50, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l2')
            tf.summary.histogram('layer2', net)
            # net = tf.layers.dense(net, 100, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
            #                       trainable=trainable, name='l3')
            # tf.summary.histogram('layer3', net)

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
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 50
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            tf.summary.histogram('layer1', net)

            net = tf.layers.dense(net, 50, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l2')
            tf.summary.histogram('layer2', net)
            # net = tf.layers.dense(net, 100, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
            #                       trainable=trainable, name='l3')
            # tf.summary.histogram('layer3', net)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, trainable=trainable)   # Q(s,a)
            tf.summary.histogram('q_value', q)
        return q

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S: s})  # shape (n_cars, 1)

    def learn(self):   # batch update
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs, ba, br, bs_ = bt[:, :self.s_dim], bt[:, self.s_dim: self.s_dim + self.a_dim], \
                          bt[:, -self.s_dim - 1: -self.s_dim], bt[:, -self.s_dim:]

        if self.output_graph:
            self.sess.run(self.a_train_op, {self.S: bs, })
            _, g_s, merged = self.sess.run([self.c_train_op, self.global_step, self.merged,], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            if g_s % 10000 == 0:
                self.writer.add_summary(merged, global_step=g_s)
        else:
            self.sess.run(self.a_train_op, {self.S: bs, })
            _, g_s = self.sess.run([self.c_train_op, self.global_step], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        if self.a_replace_counter % self.a_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        self.a_replace_counter += 1
        if self.c_replace_counter % self.c_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
        self.c_replace_counter += 1

        if g_s % self.save_model_iter == 0 and self.periodical_save:    # save model periodically
            ckpt_path = os.path.join(self.model_dir, 'DDPG.ckpt')
            save_path = self.saver.save(self.sess, ckpt_path, global_step=g_s, write_meta_graph=False)
            print("\nSave Model %s\n" % save_path)

    def store_transition(self, s, a, r, s_):
        if a.ndim < 2:
            a = a[:, np.newaxis]
        if r.ndim < 2:
            r = r[:, np.newaxis]
        transition = np.concatenate((s, a, r, s_), 1)
        p_ = self.pointer + transition.shape[0]
        if p_ <= self.memory_capacity:
            self.memory[self.pointer:p_, :] = transition
        else:
            p_ = p_ % self.memory_capacity
            self.memory[self.pointer:, :] = transition[:transition.shape[0]-p_, :]
            self.memory[:p_, :] = transition[-p_:, :]
        self.pointer = p_

    def save(self):
        ckpt_path = os.path.join(self.model_dir, 'DDPG.ckpt')
        save_path = self.saver.save(self.sess, ckpt_path, global_step=self.global_step.eval(self.sess), write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)


class DDPGPrioritizedReplay(object):
    log_dir = './log'
    save_model_iter = 100000

    def __init__(
            self,
            s_dim, a_dim, a_bound,
            a_lr=0.001, a_replace_iter=600,
            c_lr=0.001, c_replace_iter=500, gamma=0.9,
            memory_capacity=5000, batch_size=64,
            output_graph=False, restore={'load': False, 'point': 10000},
            periodical_save=False, model_dir='./model',
                 ):
        tf.reset_default_graph()

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.a_replace_iter = a_replace_iter
        self.c_replace_iter = c_replace_iter
        self.a_replace_counter = 0
        self.c_replace_counter = 0
        self.output_graph = output_graph
        self.periodical_save = periodical_save
        self.model_dir = model_dir
        self.batch_size = batch_size

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.memory_capacity = memory_capacity
        self.memory = Memory(capacity=memory_capacity)
        self.pointer = 0

        # inputs
        self.S = tf.placeholder(tf.float32, shape=[None, s_dim], name='s')
        self.R = tf.placeholder(tf.float32, [None, 1], name='r')
        self.S_ = tf.placeholder(tf.float32, shape=[None, s_dim], name='s_')

        with tf.variable_scope('Actor'):
            self.a = self._build_a_net(self.S, scope='eval_net', trainable=True)
            a_ = self._build_a_net(self.S_, scope='target_net', trainable=False)

        with tf.variable_scope('Critic'):
            q = self._build_c_net(self.S, self.a, 'eval_net', trainable=True)
            q_ = self._build_c_net(self.S_, a_, 'target_net', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('target_q'):
            target_q = self.R + gamma * q_

        with tf.variable_scope('c_loss'):
            self.abs_errors = tf.abs(target_q - q)  # for updating Sumtree
            c_loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(target_q, q))
            tf.summary.scalar('critic_loss', c_loss)

        with tf.variable_scope('C_train'):
            self.c_train_op = tf.train.AdamOptimizer(c_lr).minimize(c_loss, global_step=self.global_step, var_list=self.ce_params)

        policy_loss = - tf.reduce_mean(q)
        self.a_train_op = tf.train.AdamOptimizer(a_lr).minimize(policy_loss, global_step=self.global_step,
                                                                var_list=self.ae_params)
        self.sess = tf.Session()

        self.saver = tf.train.Saver(max_to_keep=100)
        if restore['load']:
            # all_ckpt = tf.train.get_checkpoint_state(self.model_dir, 'checkpoint').all_model_checkpoint_paths
            self.saver.restore(self.sess, restore['point'])
        else:   # clear path for saving model
            if os.path.isdir(self.model_dir):
                shutil.rmtree(self.model_dir)
            os.mkdir(self.model_dir)
            self.sess.run(tf.global_variables_initializer())

        if output_graph:
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def _build_a_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)
            net = tf.layers.dense(s, 50, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l1')
            tf.summary.histogram('layer1', net)
            net = tf.layers.dense(net, 50, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l2')
            tf.summary.histogram('layer2', net)
            # net = tf.layers.dense(net, 100, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
            #                       trainable=trainable, name='l3')
            # tf.summary.histogram('layer3', net)

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
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 50
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            tf.summary.histogram('layer1', net)

            net = tf.layers.dense(net, 50, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  trainable=trainable, name='l2')
            tf.summary.histogram('layer2', net)
            # net = tf.layers.dense(net, 100, tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
            #                       trainable=trainable, name='l3')
            # tf.summary.histogram('layer3', net)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, trainable=trainable)   # Q(s,a)
            tf.summary.histogram('q_value', q)
        return q

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S: s})  # shape (n_cars, 1)

    def learn(self):   # batch update
        tree_idx, bt, ISWeights = self.memory.sample(self.batch_size)
        bs, ba, br, bs_ = bt[:, :self.s_dim], bt[:, self.s_dim: self.s_dim + self.a_dim], \
                          bt[:, -self.s_dim - 1: -self.s_dim], bt[:, -self.s_dim:]

        if self.output_graph:
            self.sess.run(self.a_train_op, {self.S: bs, })
            _, abs_errors, g_s, merged = self.sess.run([self.c_train_op, self.abs_errors, self.global_step, self.merged,],
                                                       {self.ISWeights: ISWeights, self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            if g_s % 10000 == 0:
                self.writer.add_summary(merged, global_step=g_s)
        else:
            self.sess.run(self.a_train_op, {self.S: bs, })
            _, abs_errors, g_s = self.sess.run([self.c_train_op, self.abs_errors, self.global_step],
                                               {self.ISWeights: ISWeights, self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        for i in range(len(tree_idx)):  # update priority
            idx = tree_idx[i]
            self.memory.update(idx, abs_errors[i])

        if self.a_replace_counter % self.a_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        self.a_replace_counter += 1
        if self.c_replace_counter % self.c_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
        self.c_replace_counter += 1

        if g_s % self.save_model_iter == 0 and self.periodical_save:    # save model periodically
            ckpt_path = os.path.join(self.model_dir, 'DDPG.ckpt')
            save_path = self.saver.save(self.sess, ckpt_path, global_step=g_s, write_meta_graph=False)
            print("\nSave Model %s\n" % save_path)

    def store_transition(self, s, a, r, s_):
        if a.ndim < 2:
            a = a[:, np.newaxis]
        if r.ndim < 2:
            r = r[:, np.newaxis]
        transitions = np.concatenate((s, a, r, s_), 1)
        self.memory.store(transitions)

    def save(self):
        ckpt_path = os.path.join(self.model_dir, 'DDPG.ckpt')
        save_path = self.saver.save(self.sess, ckpt_path, global_step=self.global_step.eval(self.sess), write_meta_graph=False)
        print("\nSave Model %s\n" % save_path)


if __name__ == '__main__':
    import gym

    MAX_EPISODES = 70
    MAX_EP_STEPS = 400
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    GAMMA = 0.9  # reward discount
    TAU = 0.01  # Soft update for target param, but this is computationally expansive
    # so we use replace_iter instead
    REPLACE_ITER_A = 500
    REPLACE_ITER_C = 300
    MEMORY_CAPACITY = 7000
    BATCH_SIZE = 32

    RENDER = False
    OUTPUT_GRAPH = True
    ENV_NAME = 'Pendulum-v0'
    var = 3
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [-2, 2]
    ddpg = DDPG(state_dim, action_dim, action_bound, LR_A, REPLACE_ITER_A,
                LR_C, REPLACE_ITER_C, GAMMA, MEMORY_CAPACITY, BATCH_SIZE)
    counter = 0
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):

            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s[np.newaxis, :])[0]
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s[np.newaxis, :], a, np.array([r / 10]), s_[np.newaxis, :])

            if counter >= MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                ddpg.learn()

            counter += 1
            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -1000:
                    RENDER = True
                break