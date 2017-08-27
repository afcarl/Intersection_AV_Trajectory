import numpy as np
import math
from numba import njit, jit, vectorize
import cProfile


import timeit
from functools import partial
def timing(loop=1000, repeat=3):
    def _timing(fun):
        def wrap(*args, **kwargs):
            times = timeit.Timer(partial(fun, *args, **kwargs)).repeat(repeat=repeat, number=loop)
            min_time = min(times)/loop
            print("%f ms - rpt: %i |loop: %i |fc: %s" % (min_time * 1000, repeat, loop, fun.__name__))
            return fun(*args, **kwargs)
        return wrap
    return _timing

# @timing(loop=200, repeat=3)
def no_jit(a, b):
    idx = np.random.randint(0, 9999, b)
    return np.take(a, idx, axis=0)


# @timing(loop=200, repeat=3)
@njit('f8[:,:](f8[:,:], i8)')
def with_jit(a, batch_size):
    index = np.random.randint(0, 9999, batch_size)
    return a[index]

a = np.random.rand(99999,99)

pr = cProfile.Profile()
pr.enable()
print(np.array_equal(no_jit(a, 55), with_jit(a, 55)))
pr.disable()
pr.print_stats(sort='time')