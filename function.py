import json
import re
from collections import OrderedDict
import numpy as np


def missed_assignments(tasks, agents=None):
    return [t.missed_assignments(agents) for t in tasks]


def max_task_pressures(tasks, agents=None):
    return [max(t.relative_affinities(agents)) for t in tasks]


def task_pressures(tasks, agents=None):
    return [t.pressure(agents) for t in tasks]


def affinity_pressure(tasks, agents=None):
    return max(task_pressures(tasks, agents))


def affinity_pressure_mean(tasks, agents=None):
    return np.mean(task_pressures(tasks, agents))


def affinity_pressure_percentile(tasks, agents=None, perc=95):
    return np.percentile(task_pressures(tasks, agents), q=perc)


def load_instance(instance):
    tasks = {}
    agents = {}
    task_avail = []
    agent_avail = []
    profits = []
    assignments = {}

    task_regex = re.compile(
        r"task\((?P<name>\d+),(?P<weights>\[[\d,\s]+\]),(?P<profits>\[[\d,\s]+\]),(?P<poss_agents>\[[\d,\s]*\])\)")
    agent_regex = re.compile(r"agent\((?P<name>\d+),(?P<capacity>\d+)\)")
    assign_regex = re.compile(r"assignment\((?P<name>\d+),(?P<assigned_tasks>\[[\d,\s]*\])\)")
    avail_regex = re.compile(r"(?P<type>(task|agent)avail)\((?P<cycle>\d+),(?P<availabilities>\[[\d,\s]*\])\)")
    profit_regex = re.compile(r"profit\((?P<cycle>\d+),(?P<profits>[\d\s,\[\]]*)\).")

    for line in open(instance, 'r'):
        line = line.replace(' ', '')

        m = task_regex.match(line)
        if m:
            name = int(m.group('name'))
            weights = json.loads(m.group('weights'))
            poss_agents = json.loads(m.group('poss_agents'))
            prios = json.loads(m.group('profits'))

            assert (len(weights) == len(poss_agents))
            assert (len(prios) == len(poss_agents))
            assert (name not in tasks)

            t = Task(name, weights, prios, poss_agents)
            tasks[name] = t

        m = agent_regex.match(line)

        if m:
            name = int(m.group('name'))
            capacity = int(m.group('capacity'))

            assert (capacity > 0)
            assert (name not in agents)

            a = Agent(name, capacity)
            agents[name] = a

        m = avail_regex.match(line)

        if m and m.group('type') == 'taskavail':
            cycle = int(m.group('cycle'))
            availabilities = json.loads(m.group('availabilities'))
            task_avail.insert(cycle - 1, availabilities)
        elif m and m.group('type') == 'agentavail':
            cycle = int(m.group('cycle'))
            availabilities = json.loads(m.group('availabilities'))
            agent_avail.insert(cycle - 1, availabilities)

        m = assign_regex.match(line)

        if m:
            agent = int(m.group('name'))
            assigned_tasks = json.loads(m.group('assigned_tasks'))
            assignments[agent] = assigned_tasks

        m = profit_regex.match(line)

        if m:
            cycle = int(m.group('cycle'))
            prof = json.loads(m.group('profits'))
            profits.append((cycle, prof))

    if len(profits) > 0:
        assert (len(profits) == len(task_avail) - 1)

        for _, prof in sorted(profits, key=lambda x: x[0]):
            for t, p in zip(tasks, prof):
                tasks[t].future_profits.append(p)

    assert (len(task_avail) == len(agent_avail))
    assert (all(len(x) <= len(tasks) for x in task_avail))
    assert (all(len(x) <= len(agents) for x in agent_avail))

    return tasks, agents, task_avail, agent_avail, assignments


def all_assignable(tasks, agents):
    agent_names = [a.name for a in agents]

    for t in tasks:
        if not any((k for k in t.profits.keys() if k in agent_names)):
            return False

    return True


class Task(object):
    def __init__(self, name, weights, profits, poss_agents, future_profits=[]):
        self.name = name
        self.weights = OrderedDict(zip(poss_agents, weights))
        self.profits = OrderedDict(zip(poss_agents, profits))
        self.affinities = OrderedDict([(name, 1) for name in poss_agents])
        self.future_profits = future_profits
        self.history = []
        self.backup_profits = None

    def relative_affinities(self, agents=None):
        affs = self._filtered_affinities(agents)
        nb_affs = len(affs)
        return [float(x) / nb_affs for x in affs]

    def pressure(self, agents=None):
        affs = self._filtered_affinities(agents)
        C = len(affs)
        actual = self.affinity_sum(agents)
        ideal = self.ideal_affinity_sum(agents)
        return (actual - ideal) / C

    def affinity_sum(self, agents=None):
        return self._filtered_affinities(agents).sum()

    def ideal_affinity_sum(self, agents=None):
        C = len(self._filtered_affinities(agents))
        return C * (C + 1) / 2

    def _filtered_affinities(self, agents=None):
        if agents:
            agent_names = [a.name for a in agents]
            affs = np.array([v for k, v in self.affinities.items() if k in agent_names])
        else:
            affs = np.array(list(self.affinities.values()))

        # Scale affinities by their min., to give higher importance to completely unassigned tasks
        # Skipped, because experiments showed no benefit from this
        # Left here for potential later further evaluation & inspection
        #affs *= np.min(affs)

        return affs

    def missed_assignments(self, agents=None):
        rel_aff = np.floor(self.relative_affinities(agents))
        return np.sum(rel_aff)

    def update(self, assigned_agent, agents):
        if isinstance(agents, dict):
            agents = agents.values()

        for ag in agents:
            if ag.name in self.affinities:
                self.affinities[ag.name] += 1

        if assigned_agent:
            self.affinities[assigned_agent.name] = 1
            self.history.append(assigned_agent.name)

    def update_profit(self):
        """ Emulates test case prioritization """
        if self.backup_profits is not None:
            self.profits = OrderedDict(self.backup_profits)
            self.backup_profits = None

        if len(self.future_profits) == 0:
            return

        poss_agents = self.profits.keys()
        next_prio = self.future_profits.pop()
        prio_dict = [(pa, next_prio) for pa in poss_agents]
        self.profits = OrderedDict(prio_dict)

    def restrict_agent(self, agent_name):
        if self.backup_profits is None:
            self.backup_profits = OrderedDict(self.profits)

        del self.profits[agent_name]

    @property
    def poss_agents(self):
        return self.profits.keys()

    def __str__(self):
        weights = ",".join(map(str, self.weights.values()))
        prios = ",".join(map(str, self.profits.values()))
        poss_agents = ",".join(map(str, self.profits.keys()))
        stringrep = "task(%d,[%s],[%s],[%s])." % (self.name, weights, prios,
                                                  poss_agents)
        return stringrep


class Agent(object):
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity

    def __str__(self):
        return "agent(%d,%d)." % (self.name, self.capacity)

    def __hash__(self):
        return self.name


def matrizes(agents, tasks, pad_dummy_agent=False):
    prof_matrix = []
    affinity_matrix = []
    weight_matrix = []

    for t in tasks:
        full_profits = [0] if pad_dummy_agent else []
        full_affinities = [0] if pad_dummy_agent else []
        full_weights = [0] if pad_dummy_agent else []

        for i, a in enumerate(agents, start=1):
            if a.name in t.poss_agents:
                full_profits.append(t.profits[a.name])
                full_affinities.append(t.affinities[a.name])
                full_weights.append(t.weights[a.name])
            else:
                full_profits.append(0)
                full_affinities.append(0)
                full_weights.append(0)

        prof_matrix.append(full_profits)
        affinity_matrix.append(full_affinities)
        weight_matrix.append(full_weights)

    prof_mat = np.array(prof_matrix, dtype=int)
    aff_mat = np.array(affinity_matrix, dtype=int)
    weight_mat = np.array(weight_matrix,dtype=int)

    assert (prof_mat.shape == (len(tasks), len(agents) + pad_dummy_agent))
    assert (aff_mat.shape == (len(tasks), len(agents) + pad_dummy_agent))
    assert (weight_mat.shape == (len(tasks), len(agents) + pad_dummy_agent))

    return prof_mat, aff_mat, weight_mat