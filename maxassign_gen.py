import argparse
import os
import random
import sys
import numpy as np
from function import Task, Agent


def generate_swmod(nb_cycles, nb_tasks, nb_agents,
                   task_assignable_to, task_perc_avail, agent_perc_avail,
                   min_capacity, max_capacity, min_weight, max_weight,
                   diff_weights):
    c = np.random.randint(min_capacity, max_capacity + 1)
    capacities = np.repeat(c, nb_agents)

    if diff_weights:
        weights = np.random.randint(min_weight, max_weight + 1, (nb_tasks, nb_agents))
    else:
        weights = np.random.randint(min_weight, max_weight + 1, (nb_tasks, 1))
        weights = np.repeat(weights, nb_agents, axis=1)

    compatibility = np.random.rand(nb_tasks, nb_agents) < task_assignable_to

    initial_profits = np.random.randint(1, 101, (nb_tasks, nb_agents))
    profits = np.random.randint(1, 101, (nb_cycles-1, nb_tasks))

    initial_profits *= compatibility
    weights *= compatibility

    assert (profits.shape == (nb_cycles-1, nb_tasks))
    assert (weights.shape == (nb_tasks, nb_agents))
    assert (compatibility.shape == (nb_tasks, nb_agents))

    profits = np.round(profits).astype(int)
    weights = np.round(weights).astype(int)
    capacities = np.round(capacities).astype(int)

    agent_availability = []
    task_availability = []

    agent_missing_cycles = np.zeros(nb_agents)

    for i in range(nb_cycles):
        agent_missing_cycles = np.maximum(agent_missing_cycles-1, 0)
        prev_unavail = np.where(agent_missing_cycles > 0)[0]
        unavail = np.where(np.random.rand(nb_agents) > agent_perc_avail)[0]
        new_unavail = np.setdiff1d(unavail, prev_unavail)
        agent_missing_cycles[new_unavail] = np.random.randint(3, 8, new_unavail.shape)  # Missing 3-7 cycles

        aavail = np.setdiff1d(np.arange(nb_agents), unavail)
        agent_availability.append(aavail)

        tavail = np.where(np.random.rand(nb_tasks) < task_perc_avail)[0]
        executable_tasks = compatibility[:, aavail]
        executable_tasks = executable_tasks.any(axis=1)
        executable_tasks = executable_tasks.nonzero()[0]
        available_tasks = np.intersect1d(tavail, executable_tasks)
        task_availability.append(available_tasks)

    task_availability = np.array(task_availability)

    return initial_profits, weights, capacities, agent_availability, task_availability, profits


def generate_multiple_knapsack(nb_cycles, nb_tasks, nb_agents,
                               task_assignable_to, task_perc_avail,
                               agent_perc_avail, correlation='weakly', weight_bounds=(10, 1000)):
    m1, m2 = weight_bounds
    weights = np.random.randint(m1, m2 + 1, (1, nb_tasks))
    weights = np.repeat(weights, nb_agents, axis=0)

    if correlation == 'uncorrelated':
        profits = np.random.randint(m1, m2 + 1, (1, nb_tasks))
        profits = np.repeat(profits, nb_agents, axis=0)
    elif correlation == 'weakly':
        profits = weights + np.random.randint(-(m2 - m1) / 10, (m2 - m1) / 10 + 1, nb_tasks)
    elif correlation == 'strongly':
        profits = weights + (m2 - m1) / 10
    elif correlation == 'subsetsum':
        profits = weights
    else:
        raise Exception('Unknown correlation: %s' % correlation)

    # weights is repeated to matrix form, therefore we only sum the first row
    capacities = np.zeros(nb_agents)

    for i in range(nb_agents - 1):
        capacities[i] = np.sum(weights[i, :]) / nb_agents * ((0.6 - 0.4) * np.random.rand() + 0.4)

    capacities[-1] = 0.5 * np.sum(weights[-1, :]) - np.sum(capacities[0:-1])

    if correlation == 'subsetsum':
        total_cap = capacities.sum()
        agent_cap = int(total_cap/nb_agents)
        capacities = agent_cap * np.ones(nb_agents)

    profits = profits.T
    weights = weights.T

    # compatibility = np.random.rand(nb_tasks, nb_agents) < task_assignable_to
    # assignable = np.round(task_assignable_to * nb_agents).astype(int)

    compatibility = weights <= capacities

    assignable = np.count_nonzero(compatibility) / compatibility.size
    assignable = np.round(assignable, decimals=2)
    print(assignable)

    profits = profits * compatibility
    weights = weights * compatibility

    assert (profits.shape == (nb_tasks, nb_agents))
    assert (weights.shape == (nb_tasks, nb_agents))
    assert (compatibility.shape == (nb_tasks, nb_agents))

    profits = np.round(profits).astype(int)
    weights = np.round(weights).astype(int)
    capacities = np.round(capacities).astype(int)

    agent_availability = []
    task_availability = []

    for i in range(nb_cycles):
        aavail = np.where(np.random.rand(nb_agents) < agent_perc_avail)[0]
        agent_availability.append(aavail)

        executable_tasks = compatibility[:, aavail]
        executable_tasks = executable_tasks.any(axis=1)
        executable_tasks = executable_tasks.nonzero()[0]

        tavail = np.where(np.random.rand(nb_tasks) < task_perc_avail)[0]
        available_tasks = np.intersect1d(tavail, executable_tasks)
        task_availability.append(available_tasks)

    return profits, weights, capacities, agent_availability, task_availability, assignable


def generate_general_assignment(nb_cycles, nb_tasks, nb_agents,
                                task_assignable_to, task_perc_avail,
                                agent_perc_avail, tightness):
    # compatibility = np.random.rand(nb_tasks, nb_agents) < task_assignable_to
    assignable = np.round(task_assignable_to * nb_agents).astype(int)
    compatibility = np.zeros((nb_tasks, nb_agents))

    for i in range(nb_tasks):
        assign = np.random.choice(nb_agents, assignable, replace=False)
        compatibility[i, assign] = 1

    weights = np.random.randint(1, 1000, (nb_tasks, nb_agents)) * compatibility
    capacities = tightness * np.sum(weights, axis=0) + np.max(weights, axis=0)
    profits = np.sum(weights, axis=1) / nb_agents + 500 * np.random.rand(nb_tasks)

    profits = np.round(profits).astype(int)
    weights = np.round(weights).astype(int)
    capacities = np.round(capacities).astype(int)

    nb_avail_agents = int(agent_perc_avail * nb_agents)
    nb_avail_tasks = int(task_perc_avail * nb_tasks)
    agent_availability = np.zeros((nb_cycles, nb_avail_agents), dtype=int)
    task_availability = []

    for i in range(nb_cycles):
        agent_availability[i, :] = np.sort(np.random.choice(nb_agents, nb_avail_agents, replace=False))
        executable_tasks = compatibility[:, agent_availability[i, :]]
        executable_tasks = executable_tasks.any(axis=1)
        executable_tasks = executable_tasks.nonzero()[0]

        if len(executable_tasks) > nb_avail_tasks:
            executable_tasks = np.random.choice(executable_tasks,
                                                nb_avail_tasks, replace=False)

        task_availability.append(np.sort(executable_tasks))

    task_availability = np.array(task_availability)

    return profits, weights, capacities, agent_availability, task_availability


def write_to_file(profits, weights, capacities, agent_availability,
                  task_availability, future_profits, outfile):
    with open(outfile, 'w') as f:
        for i, c in enumerate(capacities, start=1):
            a = Agent(i, c)
            f.write('%s\n' % a)

        for i, (p, w) in enumerate(zip(profits, weights), start=1):
            poss_agents = w.nonzero()[0] + 1
            t = Task(i, w[w.nonzero()], p[p.nonzero()], poss_agents)
            f.write('%s\n' % t)

        for cycle, aa in enumerate(agent_availability, start=1):
            f.write("agentavail(%d, %s).\n" % (cycle, list(aa + 1)))

        for cycle, ta in enumerate(task_availability, start=1):
            f.write("taskavail(%d, %s).\n" % (cycle, list(ta + 1)))

        for cycle, prof in enumerate(future_profits, start=2):
            f.write("profit(%d, %s).\n" % (cycle, prof.tolist()))


def is_valid_instance(profits, weights, capacities, agent_availability, task_availability, future_profits=None):
    only_positive_integers = np.all(weights >= 0) and np.all(capacities > 0) and np.all(profits >= 0)
    an_agent_can_hold_each_comp_task = np.all(weights.max(axis=0) <= capacities)
    each_task_has_compatible_agents = np.all(weights.sum(axis=1) > 0)
    each_agent_has_compatible_tasks = np.all(weights.sum(axis=0) > 0)
    an_agent_cannot_hold_all_comp_tasks = np.all(weights.sum(axis=0) > capacities)

    unique_agents = set()
    unique_tasks = set()
    for aa, ta in zip(agent_availability, task_availability):
        unique_agents.update(aa)
        unique_tasks.update(ta)

    each_agent_is_available_at_least_once = len(unique_agents) == profits.shape[1]
    each_task_is_available_at_least_once = len(unique_tasks) == profits.shape[0]

    if future_profits is not None and len(future_profits) > 0:
        future_profit_size = future_profits.shape == (len(agent_availability)-1, profits.shape[0])
    else:
        future_profit_size = True

    return all([only_positive_integers, an_agent_can_hold_each_comp_task,
                each_task_has_compatible_agents, an_agent_cannot_hold_all_comp_tasks,
                each_agent_has_compatible_tasks, each_agent_is_available_at_least_once,
                each_task_is_available_at_least_once, future_profit_size])


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('agents', type=int)
    parser.add_argument('tasks', type=int)
    parser.add_argument('--type', choices=['swmod', 'uncorrelated',
                                           'weakly', 'strongly', 'subsetsum'], default='weakly')
    parser.add_argument('--cycles', type=int, default=0)
    parser.add_argument('--id', default='1')
    parser.add_argument('--assignable', type=float, default=0.8)
    parser.add_argument('--task_avail', type=float, default=0.8)
    parser.add_argument('--agent_avail', type=float, default=0.8)
    parser.add_argument('--min_capacity', type=int, default=21600)
    parser.add_argument('--max_capacity', type=int, default=36000)
    parser.add_argument('--min_weight', type=int, default=300)
    parser.add_argument('--max_weight', type=int, default=1800)
    parser.add_argument('--diff_weights', action='store_true', default=False,
                        help='Different weights per agent?')
    parser.add_argument('--output_dir', default='instances/')
    args = parser.parse_args(arguments)

    nb_cycles = args.cycles if args.cycles > 0 else 2 * max(args.agents, args.tasks)

    filename_tmpl = 'a%d_t%d_c%d_aa%.2f_ta%.2f_ass%.2f_%s_%s.pl'

    if args.type == 'swmod':
        nb_agents = args.agents  # 20 -- 30
        nb_tasks = args.tasks
        nb_cycles = args.cycles  # 365

        valid_instance = False
        min_weight = args.min_weight if args.min_weight else 300
        max_weight = args.max_weight if args.min_weight else 1800

        while not valid_instance:
            p, w, c, aa, ta, fp = generate_swmod(nb_cycles, nb_tasks, nb_agents, args.assignable,
                                                 args.task_avail, args.agent_avail,
                                                 args.min_capacity, args.max_capacity, min_weight, max_weight,
                                                 args.diff_weights)

            valid_instance = is_valid_instance(p, w, c, aa, ta, fp)

        filename = filename_tmpl % (nb_agents,
                                    nb_tasks,
                                    nb_cycles,
                                    args.agent_avail,
                                    args.task_avail,
                                    args.assignable,
                                    args.type[0:2],
                                    args.id)
    else:
        valid_instance = False
        min_weight = args.min_weight if args.min_weight else 10
        max_weight = args.max_weight if args.min_weight else 1000

        while not valid_instance:
            p, w, c, aa, ta, ass = generate_multiple_knapsack(nb_cycles, args.tasks, args.agents,
                                                              task_assignable_to=args.assignable,
                                                              task_perc_avail=args.task_avail,
                                                              agent_perc_avail=args.agent_avail,
                                                              correlation=args.type,
                                                              weight_bounds=(min_weight, max_weight))
            valid_instance = is_valid_instance(p, w, c, aa, ta)

        nb_tasks = len(p)
        nb_agents = len(c)
        fp = []
        filename = filename_tmpl % (nb_agents,
                                    nb_tasks,
                                    nb_cycles,
                                    args.agent_avail,
                                    args.task_avail,
                                    ass,
                                    args.type[0:2],
                                    args.id)

    assert (args.tasks == len(p))
    assert (args.agents == len(c))

    outfile = os.path.join(args.output_dir, filename)
    write_to_file(p, w, c, aa, ta, fp, outfile)


if __name__ == '__main__':
    main(sys.argv[1:])
