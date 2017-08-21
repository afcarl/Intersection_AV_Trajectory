import numpy as np
import math
from numba import njit, jit, vectorize


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

@timing(loop=200, repeat=3)
def no_jit(a, b, c):
    return b**2 - 4 * a * c

@timing(loop=200, repeat=3)
@vectorize(['float64(float64, float64, float64)'])
def with_jit(a, b, c):
    return b**2 - 4 * a * c

a = np.random.rand(9999,)
print(np.array_equal(no_jit(a, a, a), with_jit(a, a, a)))
