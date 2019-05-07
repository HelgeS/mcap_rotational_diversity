import os
import subprocess
import time

import numpy as np
import pymzn

import mulknap
from function import load_instance, matrizes


def has_floats(x):
    x_int = x.astype(int)
    return np.any((x - x_int) != 0)


class MaxAssignment(object):
    def __init__(self, timeout=60):
        self.timeout = timeout

    def optimize(self, tasks, agents, profits, directory, filename=None):
        infile = self.export_cycle(tasks, agents, profits, filename,
                                   directory)
        outfile = infile.replace('_in.pl', '_out.pl')

        if os.path.isfile('maxassign.exe'):
            cmds = ['./maxassign.exe', infile, outfile, str(self.timeout * 1000)]
        else:
            cmds = ['sicstus', '--nologo', '--noinfo', '-l', 'maxassign.pl',
                    '--goal', "go('%s', '%s', %d),halt." % (infile, outfile,
                                                            self.timeout * 1000)]
        start = time.time()
        output = subprocess.check_output(cmds)
        duration = time.time() - start

        output_lines = output.strip().splitlines()
        # timeout_flag = output_lines[-2].decode("utf-8")
        objective = int(output_lines[-1])

        _, _, _, _, assignments = load_instance(outfile)

        return objective, duration, assignments

    def export_cycle(self, tasks, agents, profits, filename=None,
                     directory='/tmp'):
        outfile = os.path.join(directory, filename)
        agent_names = [a.name for a in agents]

        with open(outfile, 'w') as f:
            for a in agents:
                f.write('%s\n' % a)

            f.write('\n')

            for t, p in zip(tasks, profits):
                prios = ",".join(map(str, p))
                weights = [str(v) for k, v in t.weights.items() if k in
                           agent_names]
                weights = ",".join(weights)
                avail_agents = [str(x) for x in t.profits.keys() if x in
                                agent_names]
                poss_agents = ",".join(avail_agents)
                stringrep = "task(%d,[%s],[%s],[%s])." % (t.name, weights,
                                                          prios, poss_agents)
                f.write('%s\n' % stringrep)

        return outfile


class MaxAssignmentMinizinc(object):
    def __init__(self, solver='cplex', timeout=60):
        self.timeout = timeout
        self.solver = solver
        if solver == 'cplex':
            self.solver = MinizincSolver(solver='cplex')
        else:
            self.solver = solver

    def optimize(self, tasks, agents, profits, directory, filename=None):
        # The model only processes ints, but some strategies might deliver floats
        multiplier = 100 if any(has_floats(p) for p in profits) else 1
        profits = [np.round(p * multiplier).astype(int) for p in profits]

        infile = self.export_cycle(tasks, agents, profits, filename, directory)

        start = time.time()
        # Timeout: CBC/Cplex use seconds, check for other solvers if needed
        output = pymzn.minizinc('maxassign.mzn', infile + '.dzn',
                                solver=self.solver, timeout=self.timeout)
        duration = time.time() - start
        objective = output[0]['objective']
        assignment_mat = output[0]['assignment']

        am = np.array(assignment_mat).reshape((len(tasks), len(agents)))
        assigned_tasks, assigned_to = np.nonzero(am)

        assignments = {}

        for (row_idx, col_idx) in zip(assigned_tasks, assigned_to):
            agent_name = agents[col_idx].name
            task_name = tasks[row_idx].name

            assert (agent_name in tasks[row_idx].poss_agents)

            if agent_name in assignments:
                assignments[agent_name].append(task_name)
            else:
                assignments[agent_name] = [task_name]

        objective /= multiplier

        return int(objective), duration, assignments

    def export_cycle(self, tasks, agents, profits, filename=None, directory='/tmp'):
        outfile = os.path.join(directory, filename)
        agent_names = [a.name for a in agents]

        with open(outfile + '.dzn', 'w') as f:
            capacities = ", ".join((str(a.capacity) for a in agents))
            f.write('capacities = [%s];\n' % capacities)

            f.write('n_agents = %d;\n' % len(agents))
            f.write('n_tasks = %d;\n' % len(tasks))

            profitlines = 'profits=[|'
            weightlines = 'weights=[|'
            compatlines = 'compat=['

            for t, prof in zip(tasks, profits):
                profitlines += '\n'
                weightlines += '\n'
                compatlines += '\n'

                avail_agents = [a for a in agent_names if a in t.poss_agents]
                profdict = {a: p for a, p in zip(avail_agents, prof)}
                full_profits = []
                full_weights = []
                compat = []

                for i, a in enumerate(agents, start=1):
                    if a.name in profdict:
                        full_profits.append(str(profdict[a.name]))
                        full_weights.append(str(t.weights[a.name]))
                        compat.append(str(i))
                    else:
                        full_profits.append('0')
                        full_weights.append('0')

                pline = ", ".join(full_profits)
                profitlines += '%s|' % pline

                wline = ", ".join(full_weights)
                weightlines += '%s|' % wline

                cline = ", ".join(compat)
                compatlines += '{ %s }, ' % cline

            f.write(profitlines + '];\n')
            f.write(weightlines + '];\n')
            f.write(compatlines + '];\n')

        return outfile


class MultipleKnapsack(object):
    def __init__(self, timeout=60):
        self.timeout = timeout

    def optimize(self, tasks, agents, profits, directory, filename=None):
        p, _, w = matrizes(agents, tasks, pad_dummy_agent=False)
        capacities = [a.capacity for a in agents]

        objective, assignment, duration = mulknap.solve(p, w, capacities)

        assignments = {}

        for i, agidx in enumerate(assignment):
            if agidx == 0:
                continue

            agent_name = agents[agidx - 1].name
            task_name = tasks[i].name

            assert (agent_name in tasks[i].poss_agents)

            if agent_name in assignments:
                assignments[agent_name].append(task_name)
            else:
                assignments[agent_name] = [task_name]

        return int(objective), duration, assignments

    def __str__(self):
        return 'mulknap'


class MinizincSolver(pymzn.Solver):
    def __init__(self, solver='cplex'):
        super().__init__(
            'linear', support_mzn=True, support_all=True, support_num=True,
            support_timeout=True, support_stats=True, support_output_mode=True
        )
        self.solver = solver

    def args(self, mzn_file, *dzn_files, data=None, timeout=None, all_solutions=False, num_solutions=None,
             output_mode='item', parallel=1, seed=0, statistics=False, **kwargs):
        """Returns the command line arguments to start the solver"""
        args = ['minizinc', '--solver', self.solver, '--output-objective']

        if mzn_file.endswith('fzn'):
            args.append(mzn_file)
        else:
            args.append(mzn_file)

            for dzn_file in dzn_files:
                args.append(dzn_file)

        if timeout:
            args.extend(['--time-limit', str(timeout * 1000)])

        return args
