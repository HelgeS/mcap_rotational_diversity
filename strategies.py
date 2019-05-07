import time

import numpy as np
import pymzn

from function import affinity_pressure, matrizes


class Strategy(object):
    def profits(self, tasks, agents):
        pass

    def mode(self):
        return ''

    def exchange(self, agents, tasks, profits, initial_assignments, objective):
        return initial_assignments, objective

    @staticmethod
    def profits_and_affs(tasks, agents):
        profits = []
        affinities = []

        for t in tasks:
            tprio, taffs = zip(
                *[(t.profits[x.name], t.affinities[x.name]) for x in agents if x.name in t.profits.keys()])
            profits.append(np.array(tprio))
            affinities.append(np.array(taffs))

        return profits, affinities


class ProfitStrategy(Strategy):
    def profits(self, tasks, agents):
        profits, affinities = self.profits_and_affs(tasks, agents)
        return profits

    def __str__(self):
        return 'profit'


class AffinityStrategy(Strategy):
    def profits(self, tasks, agents):
        _, affinities = self.profits_and_affs(tasks, agents)
        return affinities

    def __str__(self):
        return 'affinity'


class SwitchAtThresholdStrategy(Strategy):
    def __init__(self, threshold):
        self.threshold = threshold
        self.pp = 1

    def profits(self, tasks, agents):
        profits, affinities = self.profits_and_affs(tasks, agents)
        self.pp = affinity_pressure(tasks, agents)

        if self.pp < self.threshold:
            return profits
        else:
            return affinities

    def mode(self):
        if self.pp < self.threshold:
            return 'profit'
        else:
            return 'affinity'

    def __str__(self):
        return 'switch%d' % self.threshold


class ProductCombinationStrategy(Strategy):
    def profits(self, tasks, agents):
        profits, affinities = self.profits_and_affs(tasks, agents)

        profits = [prio * aff for (prio, aff) in zip(profits, affinities)]
        return profits

    def __str__(self):
        return 'productcomb'


class WeightedPartialProfits(Strategy):
    def __init__(self, individual_weights=False):
        self.individual_weights = individual_weights
        self.weights = []

    def profits(self, tasks, agents):
        profits, affinities = self.profits_and_affs(tasks, agents)

        prio_max = np.max([p.max() for p in profits])
        aff_max = np.max([a.max() for a in affinities])

        self.weights = []

        if not self.individual_weights:
            ideal_sum = np.sum([t.ideal_affinity_sum(agents) for t in tasks])
            actual_sum = np.sum([t.affinity_sum(agents) for t in tasks])
            weight = ideal_sum / actual_sum
            weight = np.minimum(weight, 1)
            self.weights.append(weight)

        values = []

        for (t, prio, aff) in zip(tasks, profits, affinities):
            aff *= np.min(aff[aff > 0])

            if self.individual_weights:
                weight = t.ideal_affinity_sum(agents) / np.sum(aff[aff > 0])  # t.affinity_sum(agents)
                weight = min(weight, 1)
                self.weights.append(weight)

            p = (weight * prio / prio_max + (1 - weight) * aff / aff_max) * 1000

            assert (0 <= weight <= 1)
            assert (np.all(p > 0))

            values.append(p.astype(int))

        return values

    def mode(self):
        return np.mean(self.weights).round(decimals=3)

    def __str__(self):
        if self.individual_weights:
            return 'wppind'
        else:
            return 'wppshared'


class LimitedAssignment(Strategy):
    def __init__(self, core_strategy):
        self.core_strategy = core_strategy

    def profits(self, tasks, agents):
        for t in tasks:
            if len(t.poss_agents) <= 1 or len(t.history) == 0:
                # We do not remove tasks by limited assignment
                continue

            possible_assignments = [(x.name, t.affinities[x.name]) for x in agents if x.name in t.poss_agents]
            possible_assignments.sort(key=lambda x: x[1])

            if len(possible_assignments) < 2:
                # Must have at least one possible assignment left
                continue

            mean_aff = np.floor(np.mean([k[1] for k in possible_assignments]))

            for name, aff in possible_assignments:
                if aff < mean_aff:
                    t.restrict_agent(name)

            # TODO Alternative formulations: < median(affinity), < mean(affinity)
            # Mean probably better as it captures outliers, median likely to cut in half
            # name_to_remove = min(possible_assignments, key=lambda k: k[1])[0]

        # Fetch updated profits + affinities from core strategy
        values = self.core_strategy.profits(tasks, agents)

        return values

    def __str__(self):
        return str(self.core_strategy) + '-limit'


class Negotiation(ProfitStrategy):
    def __init__(self, acceptance_ratio=0.6):
        self.acceptance_ratio = acceptance_ratio

    def assignment_matrix(self, agents, tasks, assignments):
        all_assigned = []
        task_pos = [t.name for t in tasks]
        x = np.zeros((len(tasks), len(agents) + 1), dtype=bool)

        for col_idx, agent_key in enumerate(sorted(assignments), start=1):
            assigned_tasks = assignments[agent_key]

            for t in assigned_tasks:
                row_idx = task_pos.index(t)
                x[row_idx, col_idx] = 1

            all_assigned.extend(assigned_tasks)

        unassigned = set([t.name for t in tasks]) - set(all_assigned)

        for t in unassigned:
            row_idx = task_pos.index(t)
            x[row_idx, 0] = 1

        return x

    def assignment_mat_to_dict(self, agents, tasks, x):
        new_assignments = {}

        for agent_idx, column in enumerate(x[:, 1:].T):
            assigned_rows = np.where(column == 1)[0]
            assigned_tasks = [tasks[r].name for r in assigned_rows]
            assert (all(agents[agent_idx].name in tasks[r].poss_agents for r in assigned_rows))
            new_assignments[agents[agent_idx].name] = assigned_tasks

        return new_assignments


class OneSwapNegotiation(Negotiation):
    def exchange(self, agents, tasks, profits, initial_assignments, objective):
        min_objective = int(objective * self.acceptance_ratio)

        print('Objective: %d / Bound: %d' % (objective, min_objective))

        candidates = []
        capacities = [0] + [a.capacity for a in agents]

        profit_matrix, aff_mat, weight_matrix = matrizes(agents, tasks, pad_dummy_agent=True)
        x = self.assignment_matrix(agents, tasks, initial_assignments)

        initial_affinities = np.sum(aff_mat * x, axis=1, keepdims=True)
        initial_profits = np.sum(profit_matrix * x, axis=1, keepdims=True)

        aff_improv = (aff_mat - initial_affinities) * (aff_mat > 0)
        aff_improv[:, [0]] -= initial_affinities

        prof_diff = (profit_matrix - initial_profits) * (profit_matrix > 0)
        prof_diff[:, [0]] -= initial_profits

        # 1. Build a list of all potential, welfare-improving exchanges
        for source_agent, affimp in enumerate(aff_improv.T):
            if source_agent == 0:
                continue  # Don't initiate from non-assigned

            pot_gains = affimp[affimp > 0]
            task_ids = np.where(affimp > 0)[0]  # row id
            sorted_order = np.argsort(affimp[affimp > 0][::-1])

            pot_gains = pot_gains[sorted_order]
            task_ids = task_ids[sorted_order]

            for dest_task, my_pot_gain in zip(task_ids, pot_gains):
                dest_agent = np.where(x[dest_task, :])[0][0]  # column id

                source_offerings = x[:, source_agent]
                dest_demand = aff_improv[:, dest_agent] > -my_pot_gain
                dest_compatible = profit_matrix[:, dest_agent] > 0  # Could also be aff_mat or weight_matrix
                potential_exchanges = np.logical_and(source_offerings, dest_demand)
                potential_exchanges = np.logical_and(potential_exchanges, dest_compatible)

                for source_task in np.where(potential_exchanges)[0]:
                    welfare_improv = my_pot_gain + aff_improv[source_task, dest_agent]
                    profit_change = prof_diff[dest_task, source_agent] + prof_diff[source_task, dest_agent]

                    assert (x[source_task, source_agent])
                    assert (x[dest_task, dest_agent])
                    assert (welfare_improv >= 0)

                    candidates.append((source_agent, source_task, dest_agent, dest_task, welfare_improv, profit_change))

        print('Tasks: %d / Candidates: %d' % (len(tasks), len(candidates)))

        # 2 Sort by 1) potential welfare improvement and 2) least profit decrease
        candidates.sort(key=lambda x: (-x[4], x[5]))

        exchanged_tasks = set()
        applied_exchanges = []

        already_exchanged = 0
        objective_bound = 0
        weight_problem = 0

        # 3. Greedily apply exchanges (this could be solved as CP/SAT or simply as a multi-pass heuristic)
        # But as long as the weight-barrier is the main failure reason, another heuristic will not help
        for (source_agent, source_task, dest_agent, dest_task, welfare_improv, profit_change) in candidates:
            if source_task in exchanged_tasks or dest_task in exchanged_tasks:
                already_exchanged += 1
                continue

            cur_weights = np.sum(weight_matrix * x, axis=0)
            new_source_weight = cur_weights[source_agent] - weight_matrix[source_task, source_agent] + weight_matrix[
                dest_task, source_agent]
            new_dest_weight = cur_weights[dest_agent] - weight_matrix[dest_task, dest_agent] + weight_matrix[
                source_task, dest_agent]

            if new_source_weight > capacities[source_agent] or new_dest_weight > capacities[dest_agent]:
                weight_problem += 1
                continue

            if (objective + profit_change) < min_objective:
                objective_bound += 1
                continue

            assert (x[source_task, source_agent])
            assert (x[dest_task, dest_agent])
            assert (not x[dest_task, source_agent])
            assert (not x[source_task, dest_agent])

            x[source_task, source_agent] = 0
            x[source_task, dest_agent] = 1
            x[dest_task, dest_agent] = 0
            x[dest_task, source_agent] = 1

            exchanged_tasks.add(source_task)
            exchanged_tasks.add(dest_task)
            applied_exchanges.append((source_agent, source_task, dest_agent, dest_task, welfare_improv, profit_change))

        new_assignments = self.assignment_mat_to_dict(agents, tasks, x)
        new_objective = np.sum(profit_matrix * x)

        assert (np.all(np.count_nonzero(x, axis=1) == 1))
        assert (np.all(np.sum(weight_matrix * x, axis=0) <= capacities))
        assert (new_objective >= min_objective)

        aff_improvement = np.sum(aff_mat * x) - np.sum(initial_affinities)
        aff_imp_perc = np.sum(aff_mat * x) / np.sum(initial_affinities) - 1.0
        objective_decrease = new_objective - objective
        print('Failure reason: Already exchanged: %d / Objective: %d / Weight: %d' % (
            already_exchanged, objective_bound, weight_problem))
        print('Changes occurred: %d / Aff. Improved: %d (%.2f) / Objective decreased: %d' % (
            len(exchanged_tasks) / 2, aff_improvement, aff_imp_perc, objective_decrease))

        return new_assignments, new_objective

    def __str__(self):
        return 'oneswap%d' % int(self.acceptance_ratio * 100)


class SolverNegotiation(Negotiation):
    def exchange(self, agents, tasks, profits, initial_assignments, objective):
        min_objective = int(objective * self.acceptance_ratio)

        print('Objective: %d / Bound: %d' % (objective, min_objective))

        candidates = set()
        capacities = [0] + [a.capacity for a in agents]

        profit_matrix, aff_mat, weight_matrix = matrizes(agents, tasks, pad_dummy_agent=True)
        x = self.assignment_matrix(agents, tasks, initial_assignments)

        initial_affinities = np.sum(aff_mat * x, axis=1, keepdims=True)
        initial_profits = np.sum(profit_matrix * x, axis=1, keepdims=True)

        aff_improv = (aff_mat - initial_affinities) * (aff_mat > 0)
        aff_improv[:, [0]] -= initial_affinities
        prof_diff = (profit_matrix - initial_profits) * (profit_matrix > 0)
        prof_diff[:, [0]] -= initial_profits

        delta_welfares = []
        delta_profits = []
        delta_weights = []
        affected_agents = []
        exchanged_tasks = []

        # 1. Build a list of all potential, welfare-improving exchanges
        for source_agent, affimp in enumerate(aff_improv.T):
            # if source_agent == 0:
            #    continue  # Don't initiate from non-assigned

            pot_gains = affimp[affimp > 0]
            task_ids = np.where(affimp > 0)[0]  # row id
            sorted_order = np.argsort(affimp[affimp > 0][::-1])

            pot_gains = pot_gains[sorted_order]
            task_ids = task_ids[sorted_order]

            for dest_task, my_pot_gain in zip(task_ids, pot_gains):
                dest_agent = np.where(x[dest_task, :])[0][0]  # column id

                source_offerings = x[:, source_agent]
                dest_demand = aff_improv[:, dest_agent] > -my_pot_gain
                dest_compatible = profit_matrix[:, dest_agent] > 0  # Could also be aff_mat or weight_matrix
                potential_exchanges = np.logical_and(source_offerings, dest_demand)
                potential_exchanges = np.logical_and(potential_exchanges, dest_compatible)

                for source_task in np.where(potential_exchanges)[0]:
                    welfare_improv = my_pot_gain + aff_improv[source_task, dest_agent]
                    profit_change = prof_diff[dest_task, source_agent] + prof_diff[source_task, dest_agent]

                    assert (x[source_task, source_agent])
                    assert (x[dest_task, dest_agent])
                    assert (welfare_improv >= 0)
                    assert (welfare_improv == aff_improv[source_task, dest_agent] + aff_improv[dest_task, source_agent])

                    ex1 = (source_agent, source_task, dest_agent, aff_improv[source_task, dest_agent],
                           prof_diff[source_task, dest_agent],
                           (-weight_matrix[source_task, source_agent], weight_matrix[source_task, dest_agent]))
                    ex2 = (dest_agent, dest_task, source_agent, aff_improv[dest_task, source_agent],
                           prof_diff[dest_task, source_agent],
                           (-weight_matrix[dest_task, dest_agent], weight_matrix[dest_task, source_agent]))
                    candidates.add(((-welfare_improv, -profit_change), ex1, ex2))

        for _, ex1, ex2 in sorted(candidates, key=lambda trans: trans[0]):
            delta_welfares.append(ex1[3])
            delta_welfares.append(ex2[3])

            delta_profits.append(ex1[4])
            delta_profits.append(ex2[4])

            delta_weights.append(list(ex1[5]))
            delta_weights.append(list(ex2[5]))

            affected_agents.append([source_agent + 1, dest_agent + 1])
            affected_agents.append([dest_agent + 1, source_agent + 1])

            exchanged_tasks.append(source_task)
            exchanged_tasks.append(dest_task)

        weight_budget = np.array(capacities) - np.sum(weight_matrix * x, axis=0)

        assert (len(delta_welfares) == len(candidates) * 2)
        assert (len(weight_budget) == len(agents) + 1)

        data = {
            'n_agents': len(agents) + 1,
            'n_tasks': len(tasks),
            'n_exchanges': len(delta_welfares),
            'profit_budget': objective - min_objective,
            'weight_budget': weight_budget,
            'delta_welfares': delta_welfares,
            'delta_profits': delta_profits,
            'delta_weights': delta_weights,
            'agents': affected_agents,
            'task_ids': exchanged_tasks
        }

        print('Exchanges: %d' % len(delta_welfares))

        if len(delta_welfares) > 0:
            pymzn.dict2dzn(data, fout='neg1.dzn')
            start = time.time()
            output = pymzn.minizinc('negotiation.mzn', solver='gecode', data=data, timeout=30)  # Not a MIP problem
            duration = time.time() - start

            sel_exchanges = output[0]['assignment']
            affinity_improvement = output[0]['objective']
            nb_exchanges = np.count_nonzero(sel_exchanges)

            print('Applied Exchanges: %d / Improvement: %d / Time: %d' % (nb_exchanges, affinity_improvement, duration))

            if nb_exchanges == 0:
                return initial_assignments, objective

            for ex_id in np.where(sel_exchanges)[0]:
                task_id = exchanged_tasks[ex_id]
                source_agent, dest_agent = affected_agents[ex_id]
                source_agent, dest_agent = source_agent - 1, dest_agent - 1

                assert (x[task_id, source_agent])
                assert (not x[task_id, dest_agent])

                x[task_id, source_agent] = 0
                x[task_id, dest_agent] = 1

            new_assignments = self.assignment_mat_to_dict(agents, tasks, x)
            new_objective = np.sum(profit_matrix * x)

            assert (np.sum(aff_mat * x) == (np.sum(initial_affinities) + affinity_improvement))
            assert (np.all(np.count_nonzero(x, axis=1) == 1))
            assert (np.all(np.sum(weight_matrix * x, axis=0) <= capacities))
            assert (new_objective >= min_objective)

            return new_assignments, new_objective
        else:
            return initial_assignments, objective

    def __str__(self):
        return 'exchange%d' % int(self.acceptance_ratio * 100)


STRATEGY_MAPPING = {
    'profit': ProfitStrategy,
    'affinity': AffinityStrategy,
    'switch': SwitchAtThresholdStrategy,
    'productcomb': ProductCombinationStrategy,
    'wpp': WeightedPartialProfits,
    'negotiation': OneSwapNegotiation,
    'exchange': SolverNegotiation
}
