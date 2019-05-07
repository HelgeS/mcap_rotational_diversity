from __future__ import division

import argparse
import os

import numpy as np

import copy

import function
import strategies
from function import load_instance, all_assignable
from problem import MaxAssignmentMinizinc, MultipleKnapsack


def main(instance, strategy, problem, output_dir):
    # Load instance
    tasks, agents, task_availability, agent_availability, _ = load_instance(instance)
    instance_name, _ = os.path.splitext(os.path.basename(instance))

    if str(problem) != 'mulknap':
        file_affix = '%s_%s' % (instance_name, strategy)
    else:
        file_affix = '%s_mulknap' % instance_name

    log_file = open(os.path.join(output_dir, '%s_log.csv' % file_affix), 'w')
    log_header = ['instance', 'strategy', 'mode', 'cycle', 'objective',
                  'profit', 'affinity', 'pressure_max', 'pressure_mean',
                  'total_pressure_max', 'total_pressure_mean', 'assigned',
                  'utilization', 'agents', 'tasks', 'timeout']
    log_template = ';'.join(('{%s}' % x for x in log_header))
    log_header_line = ';'.join(log_header)
    log_file.write('%s\n' % log_header_line)
    print(log_header_line)

    assignment_file = open(os.path.join(output_dir, '%s_assignment.csv' % file_affix), 'w')
    assignment_header = ';'.join(['instance', 'strategy', 'cycle'] + [str(t) for t in tasks])
    assignment_file.write('%s\n' % assignment_header)

    for i, (task_avail, agent_avail) in enumerate(zip(task_availability,
                                                      agent_availability),
                                                  start=1):
        for x in task_avail:
            tasks[x].update_profit()

        cycle_tasks = [tasks[x] for x in task_avail]
        cycle_agents = [agents[x] for x in agent_avail]

        assert (all_assignable(cycle_tasks, cycle_agents))

        profits = strategy.profits(cycle_tasks, cycle_agents)
        filename = '%s_%d_in.pl' % (file_affix, i)
        cap_objective, solver_duration, cap_assignments = problem.optimize(cycle_tasks,
                                                                           cycle_agents,
                                                                           profits,
                                                                           output_dir,
                                                                           filename=filename)

        # Negotation phase of two-step strategies, other strategies return the input
        # Except the negotiation experiment, no strategy uses this
        assignments, objective = strategy.exchange(cycle_agents, cycle_tasks,
                                                   profits,
                                                   cap_assignments,
                                                   cap_objective)

        all_assigned = []
        prio = 0
        aff = 0
        utilization = []

        for agent_name, assigned_tasks in assignments.items():
            assigned_weight = 0

            for t in assigned_tasks:
                prio += tasks[t].profits[agent_name]
                aff += tasks[t].affinities[agent_name]
                assigned_weight += tasks[t].weights[agent_name]
                tasks[t].update(agents[agent_name], agents)

            utilization.append(float(assigned_weight) / agents[agent_name].capacity)
            all_assigned.extend(assigned_tasks)

        unassigned = set([t.name for t in cycle_tasks]) - set(all_assigned)

        for t in unassigned:
            tasks[t].update(None, agents)

        pp_max = function.affinity_pressure(cycle_tasks, cycle_agents)
        pp_mean = function.affinity_pressure_mean(cycle_tasks, cycle_agents)
        total_pp_max = function.affinity_pressure(tasks.values())
        total_pp_mean = function.affinity_pressure(tasks.values())
        perc_assigned = len(all_assigned) / len(cycle_tasks)

        log_dict = {
            'instance': instance_name,
            'strategy': strategy,
            'mode': strategy.mode() if not (str(strategy) == 'profit' and str(problem) == 'mulknap') else 'mulknap',
            'cycle': i,
            'objective': objective,
            'profit': prio,
            'affinity': aff,
            'pressure_max': np.round(pp_max, decimals=2),
            'pressure_mean': np.round(pp_mean, decimals=2),
            'total_pressure_max': np.round(total_pp_max, decimals=2),
            'total_pressure_mean': np.round(total_pp_mean, decimals=2),
            'assigned': np.round(perc_assigned, decimals=2),
            'utilization': np.round(np.mean(utilization), decimals=2),
            'agents': len(cycle_agents),
            'tasks': len(cycle_tasks),
            'timeout': np.round(solver_duration, decimals=2)
        }

        assert (all(x in log_dict for x in log_header))

        log_entry = log_template.format(**log_dict)
        print(log_entry)
        log_file.write('%s\n' % log_entry)

        assignment_line = [instance_name, str(strategy), str(i)]

        for t in tasks.values():
            if t.name in all_assigned:
                assignment_line.append(str(t.history[-1]))
            elif t.name in unassigned:
                assignment_line.append('0')
            else:  # unavailable
                assignment_line.append('-1')

        assignment_file.write('%s\n' % ';'.join(assignment_line))

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instance')
    parser.add_argument('strategy',
                        choices=strategies.STRATEGY_MAPPING.keys())
    parser.add_argument('-p', '--problem', choices=['max_assignment', 'mulknap'],
                        default='max_assignment')
    parser.add_argument('-t', '--threshold',
                        help='Affinity Pressure Threshold (used with strategies adaptive and switch)',
                        type=float, default=3)
    parser.add_argument('--limit-assignments', action='store_true',
                        help='Limited assignment, disallow prev. agents')
    parser.add_argument('--timeout', type=int, default=60,
                        help='CP solver timeout (in s)')
    parser.add_argument('-o', '--output', default='results')
    parser.add_argument('--ind-weights', action='store_true', help='Use Individual Weights for WPP strategy')
    args = parser.parse_args()

    if args.strategy == 'switch':
        strategy = strategies.STRATEGY_MAPPING[args.strategy](args.threshold)
    elif args.strategy == 'wpp':
        strategy = strategies.STRATEGY_MAPPING[args.strategy](args.ind_weights)
    else:
        strategy = strategies.STRATEGY_MAPPING[args.strategy]()

    if args.limit_assignments:
        strategy = strategies.LimitedAssignment(strategy)

    if args.problem == 'max_assignment':
        problem = MaxAssignmentMinizinc(timeout=args.timeout)
    elif args.problem == 'mulknap':
        problem = MultipleKnapsack()
        assert (args.strategy == 'profit')

    main(args.instance, strategy, problem, args.output)
