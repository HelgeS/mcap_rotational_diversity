from ctypes import cdll, c_int, byref
import os.path
import numpy as np
import time

"""
Requires mulknap.so, which can be compiled from the sources at
http://www.diku.dk/~pisinger/codes.html In `mulknap.c` it might be necessary to
remove the defines in the section `timing routines` (around line 289)
"""

__all__ = ['test', 'solve']

mulknap_path = os.path.join(os.path.dirname(__file__), 'mulknap.so')
lib = cdll.LoadLibrary(mulknap_path)


def test():
    n = 5
    m = 2
    n_ints = c_int * n
    p = n_ints(2, 2, 2, 2, 2)
    w = n_ints(1, 1, 4, 1, 1)
    x = n_ints()
    m_ints = c_int * m
    c = m_ints(3, 3)
    z = lib.mulknap(n, m, p, w, byref(x), c)

    assert(z == 8)
    assert(list(x) == [1, 2, 0, 1, 1])
    print('All systems go')


def solve(profits, weights, capacities):
    assert (profits.shape == weights.shape)
    assert (profits.shape[1] == len(capacities))
    assert (all((len(np.unique(r[r > 0])) == 1 for r in profits)))
    assert (all((len(np.unique(r[r > 0])) == 1 for r in weights)))

    n, m = profits.shape

    m_ints = c_int * m
    n_ints = c_int * n

    agent_order = np.argsort(capacities) #[::-1]
    agent_names = np.arange(1, len(capacities)+1)[agent_order]

    capacities = np.array(capacities)

    p = n_ints(*profits.max(axis=1))
    w = n_ints(*weights.max(axis=1))
    assignment = n_ints()
    c = m_ints(*capacities[agent_order])

    start = time.time()
    objective = lib.mulknap(n, m, p, w, byref(assignment), c)
    duration = time.time() - start

    adj_assignment = [agent_names[x-1] if x > 0 else 0 for x in assignment]

    return objective, adj_assignment, duration


if __name__ == '__main__':
    test()
