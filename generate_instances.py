#!/usr/bin/env python
import itertools
import maxassign_gen

"""
Multi-Cycle Multiple Knapsack Problem
cite:Fukanaga2011
2 - 2.5 - 3 - 4 - 5 - 6 - 10
30/60, 30/75, 15/45, 12/48, 15/75, 10/60, 10/100
"""
mcmkp_agents_tasks = [(30, 75), (15, 45), (12, 48)]  # (10, 100)]
assignable = [1.0]  # , 0.75, 0.5]
availability = [1.0, 0.75]
iterations = range(1, 1 + 1)
correlations = ['weakly', 'uncorrelated']

tcsa_agents_tasks = [(20, 750), (20, 1500), (20, 3000), (30, 3000)]

if __name__ == '__main__':
    iterator = itertools.product(mcmkp_agents_tasks, correlations, assignable,
                                 availability, availability, iterations)

    # MCMKP
    for (ag, ta), corr, ass, aavail, tavail, curiter in iterator:
        args = [ag, ta, '--cycles', 3 * ta, '--assignable', ass, '--task_avail',
                tavail, '--agent_avail', aavail, '--id', curiter, '--type', corr]
        maxassign_gen.main(map(str, args))
        # break

    # Test Case Selection and Assignment
    for (ag, ta), curiter in itertools.product(tcsa_agents_tasks, iterations):
        args = [ag, ta, '--cycles', 365, '--assignable', 0.6, '--task_avail', 0.9, '--agent_avail', 0.6, '--id',
                curiter,
                '--type', 'swmod', '--min_capacity', 36000, '--max_capacity', 36000, '--min_weight', 60, '--max_weight',
                1260]
        maxassign_gen.main(map(str, args))

    # MCMSSP
    for (ag, ta), corr, aavail, tavail, curiter in itertools.product([(1, 20), (5, 20)],
                                                                     ['subsetsum'],
                                                                     [1.0],
                                                                     [0.75, 0.5],
                                                                     range(1, 4)):
        args = [ag, ta, '--cycles', 100, '--task_avail',
                tavail, '--agent_avail', aavail, '--id', curiter, '--type', corr,
                '--min_weight', 10, '--max_weight', 100]
        maxassign_gen.main(map(str, args))
        # break
