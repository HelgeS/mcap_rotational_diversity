import os
import random
import argparse

from function import Agent, Task


def read_mknap1(inpath):
    with open(inpath, 'r') as fin:
        numbers = [int(x.strip()) for x in fin.read().strip().split() if x.isdigit()]

        instancenb = numbers[0]
        idx = 1

        for _ in range(1, instancenb+1):
            tasknb, agentnb = numbers[idx:idx+2]
            instance_length = 3 + tasknb + agentnb*tasknb + agentnb
            agentnb, tasknb, capacities, profits, weights = split_instance(numbers[idx:idx+instance_length])
            idx += instance_length
            yield agentnb, tasknb, capacities, profits, weights


def split_instance(numbers):
    tasknb, agentnb, opt_value = numbers[0:3]
    profits = numbers[3:3+tasknb]

    weights = []
    baseidx = 3+tasknb

    for _ in range(agentnb):
        weights.append(numbers[baseidx:baseidx+tasknb])
        baseidx += tasknb

    capacities = numbers[baseidx:baseidx+agentnb]

    assert(baseidx+agentnb == len(numbers))
    assert(len(weights) == agentnb)

    return agentnb, tasknb, capacities, profits, weights


def read_mknap2(inpath):
    fin = open(inpath, 'r')

    numbers = [int(x.strip()) for x in fin.read().strip().split() if x.isdigit()]
    agentnb, tasknb = numbers[0:2]
    profits = numbers[2:2+tasknb]
    capacities = numbers[2+tasknb:2+tasknb+agentnb]

    weights = []
    baseidx = 2+tasknb+agentnb

    for _ in range(agentnb):
        weights.append(numbers[baseidx:baseidx+tasknb])
        baseidx += tasknb

    fin.close()

    yield agentnb, tasknb, capacities, profits, weights


def convert(inpath, outdir, cycles, task_availability, agent_availability, iteration):
    basename, ext = os.path.splitext(os.path.basename(inpath))

    if ext.lower() == '.txt':
        read = read_mknap1
    elif ext.lower() == '.dat':
        read = read_mknap2
    else:
        raise Exception('Unsupported File Format')

    for cnt, (agentnb, tasknb, capacities, profits, weights) in enumerate(read(inpath), start=1):
        outname = 'cb_%d_%d_%d_%d_%.2f_%.2f_%d.pl' % (agentnb, tasknb, cnt,
                                                      cycles,
                                                      agent_availability,
                                                      task_availability,
                                                      iteration)
        outpath = os.path.join(outdir, outname)
        out = open(outpath, 'w')

        agents = {}
        for values in enumerate(capacities, start=1):
            out.write('agent(%d,%d).\n' % values)
            agents[values[0]] = Agent(*values)

        task_iterator = enumerate(zip(profits, zip(*weights)), start=1)
        tasks = {}
        for taskid, (p, task_weights) in task_iterator:
            wl, pa = zip(*[(w, i) for i, w in enumerate(task_weights, start=1) if w > 0])
            pl = [p] * len(pa)
            out.write('task(%d,%s,%s,%s).\n' % (taskid, list(wl), pl, list(pa)))
            tasks[taskid] = Task(taskid, list(wl), pl, list(pa))

        for idx in range(1, cycles+1):
            agent_avail = random.sample(range(1, agentnb+1),
                                        k=int(agent_availability*agentnb))
            agent_avail_names = [agents[x].name for x in agent_avail]
            executable_tasks = [t for t in tasks.values() if any(ta for ta in t.poss_agents if ta in agent_avail_names)]

            no_avail_tasks = int(task_availability*tasknb)

            if len(executable_tasks) > no_avail_tasks:
                executable_tasks = random.sample(executable_tasks, k=no_avail_tasks)

            task_avail = sorted([t.name for t in executable_tasks])

            out.write('agentavail(%d,%s).\n' % (idx, sorted(agent_avail)))
            out.write('taskavail(%d,%s).\n' % (idx, sorted(task_avail)))

        out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--cycles', type=int, default=30)
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--task_avail', type=float, default=0.8)
    parser.add_argument('--agent_avail', type=float, default=0.8)
    parser.add_argument('-o', '--output_dir', default='instances/')
    args = parser.parse_args()

    for infile in args.infiles:
        for i in range(args.iter):
            convert(infile, args.output_dir, args.cycles,
                    task_availability=args.task_avail,
                    agent_availability=args.agent_avail,
                    iteration=i+1)
