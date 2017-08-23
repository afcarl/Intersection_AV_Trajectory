import numpy as np


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(
            capacity,
            dtype=[('s', np.float32, (s_dim,)), ('a', np.float32, (a_dim,)),
                   ('r', np.float32, (1,)), ('s_', np.float32, (s_dim,))])  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, s, a, r, s_):
        for d in zip(s, a, r, s_):
            tree_idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = d  # update data_frame
            self.update(tree_idx, p)  # update tree_frame, add this p (maxp) for all transitions

            self.data_pointer += 1
            if self.data_pointer >= self.capacity:  # replace when exceed the capacity
                self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
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
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, data_idx

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.00001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, batch_size, s_dim, a_dim):
        self.tree = SumTree(capacity, s_dim, a_dim)
        self.batch_size = batch_size
        self.btree_idx, self.ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1), dtype=np.float32)
        self.bdata_idx = np.empty_like(self.btree_idx)
        self.b_memory = np.zeros(
            batch_size,
            dtype=[('s', np.float32, (s_dim,)), ('a', np.float32, (a_dim,)),
                   ('r', np.float32, (1,)), ('s_', np.float32, (s_dim,))])

    def store(self, s, a, r, s_):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:  # add maxp for all transitions
            max_p = self.abs_err_upper
        self.tree.add(max_p, s, a, r, s_)   # set the max p for new p

    def sample(self):
        pri_seg = self.tree.total_p / self.batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(self.batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            self.btree_idx[i], self.bdata_idx[i] = self.tree.get_leaf(v)

        np.take(self.tree.tree, self.btree_idx, axis=0, out=self.ISWeights[:, 0])
        np.power(self.ISWeights / self.tree.total_p / min_prob, -self.beta, out=self.ISWeights)
        np.take(self.tree.data, self.bdata_idx, axis=0, out=self.b_memory)
        return self.btree_idx, self.b_memory, self.ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = ps = abs_errors[:]
        np.minimum(abs_errors, self.abs_err_upper, out=clipped_errors)
        np.power(clipped_errors, self.alpha, out=ps)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)