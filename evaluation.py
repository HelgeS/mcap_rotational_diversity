import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from itertools import cycle
from collections import Counter
import os
import pickle
import re
import sys
import function


def figsize_column(scale, height_ratio=1.0):
    fig_width_pt = 433  # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def figsize_text(scale, height_ratio=1.0):
    fig_width_pt = 433  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": figsize_column(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}

sns.set_style("whitegrid", pgf_with_latex)
sns.set_context("paper")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

LABELS = {
    'strategy': 'Strategy',
    'pressure_max': 'Max. Affinity Pressure',
    'instance': 'Instance',
    'first_assigned': 'First Assigned',
    'rotation': 'Full Rotations',
    'fullrotations': 'Rotational Diversity',
    'rel_profit': 'Profit (% of FOP)',
    'nb_agents': 'Agents',
    'nb_tasks': 'Tasks',
    'agentavail': 'Avail. Agents',
    'taskavail': 'Avail. Tasks'
}

STRATEGY_LONG = {
    'profit': 'Profit',
    'affinity': 'Affinity',
    'switch3': 'Switch@3',
    'switch2': 'Switch@2',
    'wppshared': 'WPP/s',
    'wppind': 'WPP',
    'productcomb': 'Product Combination'
}

STRATEGY_SHORT = {
    'profit': 'FOP',
    'affinity': 'FOA',
    'switch1': 'OS/1',
    'switch2': 'OS/2',
    'switch3': 'OS/3',
    'switch4': 'OS/4',
    'switch10': 'OS/10',
    'switch20': 'OS/20',
    'switch30': 'OS/30',
    'switch40': 'OS/40',
    'wppshared': 'WPP/s',
    'wppind': 'WPP',
    'productcomb': 'PC',
    'profit-limit': 'FOP',
    'affinity-limit': 'FOA',
    'switch1-limit': 'OS/1',
    'switch2-limit': 'OS/2',
    'switch3-limit': 'OS/3',
    'switch4-limit': 'OS/4',
    'switch10-limit': 'OS/10',
    'switch20-limit': 'OS/20',
    'switch30-limit': 'OS/30',
    'switch40-limit': 'OS/40',
    'wppshared-limit': 'WPP/s',
    'wppind-limit': 'WPP',
    'productcomb-limit': 'PC'
}

FULL_LABELS = {**LABELS, **STRATEGY_LONG}

instance_cache = {}


def read_results(log_filename, ignore_errors=False):
    try:
        assign_filename = log_filename.replace('_log.csv', '_assignment.csv')
        instance = load_instance(log_filename)

        log_df = read_csv(log_filename, ignore_errors)
        assign_df = read_csv(assign_filename, ignore_errors)
        stats_df = assignment_statistics(assign_df, instance)

        nb_tasks = len(instance['tasks'])
        nb_agents = len(instance['agents'])
        agentavail = instance['aa_perc']
        taskavail = instance['ta_perc']

        log_df['nb_tasks'] = nb_tasks
        log_df['nb_agents'] = nb_agents
        log_df['agentavail'] = agentavail
        log_df['taskavail'] = taskavail

        assign_df['nb_tasks'] = nb_tasks
        assign_df['nb_agents'] = nb_agents
        assign_df['agentavail'] = agentavail
        assign_df['taskavail'] = taskavail

        stats_df['nb_tasks'] = nb_tasks
        stats_df['nb_agents'] = nb_agents
        stats_df['agentavail'] = agentavail
        stats_df['taskavail'] = taskavail

        return log_df, assign_df, stats_df
    except:
        print(log_filename)
        raise


def read_csv(filename, ignore_errors=False):
    try:
        return pd.read_csv(filename, sep=';')
    except Exception as e:
        print(filename, e)

        if not ignore_errors:
            raise


def identify_ignored_tasks(no_tasks, outfiles):
    executions = {i: [] for i in range(1, no_tasks + 1)}

    for of in sorted(outfiles):
        cycleid = int(of.rsplit('_', 2)[-2])
        _, _, _, _, assignments = function.load_instance(of)

        for assigned in assignments.values():
            for taskid in assigned:
                executions[taskid].append(cycleid)

    ignored_tasks = [taskid for taskid, execs in executions.items() if len(execs) == 0]
    return ignored_tasks


def limitation(df, ax=None):
    gdf = df[['instance', 'assigned', 'utilization']].groupby('instance', as_index=False).max()
    task_limited = gdf[gdf.assigned > 0.99]
    agent_limited = gdf[gdf.utilization > 0.99]
    print('Task-limited: %d (%.2f)\n' % (len(task_limited), len(task_limited) / len(gdf) * 100))
    print('Agent-limited: %d (%.2f)\n' % (len(agent_limited), len(agent_limited) / len(gdf) * 100))
    print(gdf)


def assignment_statistics(df, instance):
    task_columns = df.columns[~df.columns.isin(['instance', 'strategy', 'cycle', 'run'])]
    tdf = df[task_columns]

    # Cycle in which the task was first assigned to any agent; does not consider unavailability
    first_assigned = tdf.apply(lambda s: df.cycle[s.gt(0).idxmax()] if s.gt(0).any() else np.nan)  # - s.eq(-1).sum())
    # first_assigned /= tdf.shape[0]

    nb_actual_assignments = (tdf > 0).sum()
    nb_possible_assignments = (tdf >= 0).sum()
    nb_seen_agents = tdf[tdf > 0].nunique()

    wm = instance['weight_mat']
    nb_comp_agents = (wm[:, 1:] > 0).sum(axis=1)

    # DF which relative frequencies of assign. between tasks and agents
    # Rows: Agents (-1: Task unavailable, 0: Unassigned), Columns: Tasks
    # NaN: Assignment not possible
    freq = pd.DataFrame(index=np.arange(-1, wm.shape[1]))
    freq = freq.join(tdf.apply(lambda x: x.value_counts(normalize=True)))

    wm2 = np.zeros((wm.shape[1] + 1, wm.shape[0]))
    wm2[1:, :] = wm.T

    freq[freq.isnull() & (wm2 > 0)] = 0

    # How even are tasks distributed? -> mean absolute deviation of assignment distribution
    stddev = freq[freq.index > 0].mad()
    stddev[stddev.isnull()] = 0

    # Max./Avg./Min. no. of cycles between assignment to the same agent
    cycle_stats = tdf.apply(lambda s: pd.Series(assignment_cycle_stats(s)))
    cycle_stats = cycle_stats.T

    assert (cycle_stats.shape == (len(task_columns), 3))

    # Full Rotations: Min. no. of assignments to one agent
    def fullrotations(s):
        sf = s[s > 0]

        if len(sf) > 0:
            return pd.Series(Counter(sf).most_common()[-1][1])
        else:
            return pd.Series(0)

    fullrotations = tdf.apply(fullrotations) * (nb_seen_agents == nb_comp_agents)

    statsdf = np.array([task_columns,
                        cycle_stats[0] / nb_comp_agents,
                        cycle_stats[1] / nb_comp_agents,
                        cycle_stats[2] / nb_comp_agents,
                        first_assigned,
                        nb_actual_assignments / nb_possible_assignments,
                        stddev,
                        nb_actual_assignments > 0,
                        nb_seen_agents / nb_comp_agents,
                        nb_comp_agents,
                        nb_actual_assignments,
                        fullrotations.T[0]],
                       dtype=float)
    statsdf = pd.DataFrame(statsdf.T, columns=['task', 'max_cycles', 'avg_cycles', 'min_cycles', 'first_assigned',
                                               'rel_assignments', 'distribution', 'used_tasks', 'used_agents',
                                               'nb_compatible_agents', 'total_assignments', 'fullrotations'])

    if 'run' in df.columns:
        inst_name, strategy_name = df['run'][0].rsplit('_', 1)
        statsdf['instance'] = inst_name
        statsdf['strategy'] = strategy_name
    else:
        statsdf['instance'] = df['instance'][0]
        statsdf['strategy'] = df['strategy'][0]

    return statsdf


def assignment_cycle_stats(series):
    """
    Evaluates the frequency by which a task is assigned to its compatible agents.
    :param series:
    :return:
    a) Max. no. of cycles until the same agent is used again
    b) Avg. no. of cycles until the same agent is used again
    c) Min. no. of cycles until the same agent is used again
    """
    bins = series.tolist()
    dists = []
    seen = set()

    for i, b in enumerate(bins, start=1):
        if b <= 0:
            continue

        if b in bins[i:]:
            dists.append(bins[i:].index(b) + 1)
        elif len(bins[i:]) > 0:
            dists.append(len(bins[i:]))

        seen.add(b)

    if len(dists) > 0:
        return np.max(dists), np.mean(dists), np.min(dists)
    else:
        return len(series), len(series), len(series)


def load_instance(log_file):
    log_base = os.path.basename(log_file)
    instance_name = log_base.rsplit('_', 2)[0]

    if instance_name in instance_cache:
        return instance_cache[instance_name]

    instance_file = os.path.join('instances', instance_name + '.pl')
    t, a, ta, aa, _ = function.load_instance(instance_file)
    pm, am, wm = function.matrizes(a.values(), t.values(), pad_dummy_agent=True)

    m = re.match('a\d+_t\d+_c\d+_aa([\d.]+)_ta([\d.]+)_ass([\d.]+)_(\w+)_(\w+)', instance_name)
    aa_perc = float(m.group(1))
    ta_perc = float(m.group(2))
    assignable = float(m.group(3))

    instance = {
        'name': instance_name,
        'tasks': t,
        'agents': a,
        'taskavail': ta,
        'agentavail': aa,
        'profit_mat': pm,
        'aff_mat': am,
        'weight_mat': wm,
        'aa_perc': aa_perc,
        'ta_perc': ta_perc,
        'assignable': assignable
    }

    instance_cache[instance_name] = instance

    return instance


def plot_rel_profit_boxplots(logdf, filename=None):
    _, ax = plt.subplots(figsize=figsize_column(1.0, height_ratio=0.8))
    grouped_logdf = logdf[['strategy', 'nb_agents', 'agentavail', 'taskavail', 'nb_tasks', 'profit']].groupby(
        ['strategy', 'nb_agents', 'nb_tasks', 'agentavail', 'taskavail'], as_index=False).sum()
    grouped_logdf['rel_profit'] = grouped_logdf.groupby(
        ['nb_agents', 'nb_tasks', 'agentavail', 'taskavail']).profit.transform(
        lambda x: x / x.max())
    del grouped_logdf['profit']

    grouped_logdf['rel_profit'] *= 100
    grouped_logdf = grouped_logdf[grouped_logdf.strategy != 'profit']
    grouped_logdf = grouped_logdf.rename(columns={**LABELS, **STRATEGY_SHORT})
    grouped_logdf = grouped_logdf.replace(to_replace=STRATEGY_SHORT.keys(), value=STRATEGY_SHORT.values())

    sns.boxplot(x='Strategy', y='Profit (% of FOP)', data=grouped_logdf, ax=ax, linewidth=1)
    # ax.set_ylabel("Profit (% of FOP)")
    # ax.set_xlabel("Strategy")
    ax.set_ylim(top=100)

    if filename:
        plt.savefig(filename, dpi=500, bbox_inches='tight', pad_inches=0)


def plot_timeline(df, values, ax=None):
    n_strategies = len(df.strategy.unique())
    colors = sns.color_palette(n_colors=n_strategies)

    if n_strategies == 1:
        colors = colors[0]

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    for v in values:
        pdf = df.pivot(index='cycle', columns='strategy', values=v)

        if len(values) > 1:
            labels = {s: '%s (%s)' % (s.capitalize(), v.capitalize()) for s in
                      pdf.columns}
        else:
            labels = {s: s.capitalize() for s in pdf.columns}

        pdf = pdf.rename(columns=labels)
        pdf.plot(ax=ax, title=v.capitalize(), linestyle=next(linecycler), color=colors)
        ax.legend(ncol=3)


def plot_profit_bars(logdf, filename=None, ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=figsize_column(1.0))

    grouped_logdf = logdf[['strategy', 'nb_agents', 'nb_tasks', 'profit']].groupby(
        ['strategy', 'nb_agents', 'nb_tasks'], as_index=False).sum()
    grouped_logdf['rel_profit'] = grouped_logdf.groupby(['nb_agents', 'nb_tasks']).profit.transform(
        lambda x: x / x.max() * 100)
    grouped_logdf['scenario'] = grouped_logdf.apply(lambda x: '{}/{}'.format(x['nb_agents'], x['nb_tasks']), axis=1)
    # grouped_logdf = grouped_logdf.set_index(['nb_agents', 'nb_tasks'])
    sns.barplot(x='scenario', y='rel_profit', hue='strategy', data=grouped_logdf)
    ax.legend(loc=9, bbox_to_anchor=(0.5, 1.3), ncol=2)
    ax.set_xlabel('Scenario [Agents/Tasks]')
    ax.set_ylabel('Rel. Profit [%]')
    ax.set_ylim([0, 100])

    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')


def plot_timeout_bars(df, ax=None):
    pdf = df[['strategy', 'timeout']]
    pdf.loc[:, 'timeout'] = pdf['timeout'] >= 30
    pdf = pdf.groupby('strategy', as_index=False).sum()
    pdf['timeout'] = pdf['timeout'] / df.cycle.max() * 100
    sns.barplot(x='strategy', y='timeout', estimator=sum, ci=None, data=pdf,
                ax=ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Solver Timeouts [%]')
    ax.set_title('Solver Timeouts [%]')


def plot_bars(df, column, agg=np.mean, ax=None):
    pretty_title = column.replace('_', ' ').title()
    pdf = df[['strategy', column]]
    # pdf[column] *= 100
    sns.barplot(x='strategy', y=column, estimator=agg, ci=None, data=pdf, ax=ax)
    # ax.set_ylim(0, 100)
    ax.set_title(pretty_title)
    # ax.set_ylabel('{} [%]'.format(pretty_title))


def plot_boxplot(df, column, ax=None):
    pretty_title = column.replace('_', ' ').title()
    pdf = df[['strategy', column]]
    # pdf[column] *= 100
    sns.boxplot(x='strategy', y=column, data=pdf, ax=ax)
    # ax.set_ylim(0, 100)
    ax.set_title(pretty_title)
    # ax.set_ylabel('{} [%]'.format(pretty_title))


def plot_strategy_front(df, order=None, ax=None):
    pdf = df[['strategy', 'profit', 'pressure_max']].groupby('strategy', as_index=False).agg(
        {'profit': 'mean', 'pressure_max': 'mean'})
    pdf = pdf.sort_values(by=['profit', 'pressure_max'],
                          ascending=[False, True])
    pdf = pdf.rename(columns=FULL_LABELS)
    pdf = pdf.replace(to_replace=FULL_LABELS.keys(), value=FULL_LABELS.values())

    pdf.plot.scatter(x='Profit', y='Max. Affinity Pressure', c=sns.color_palette(), ax=ax, s=50)

    for s, p, a in zip(pdf['Strategy'], pdf['Profit'], pdf['Max. Affinity Pressure']):
        txt = ax.annotate(s, (p, a))
        txt.set_rotation(25)

    ax.set_xlabel('$\sum Profit$')
    ax.set_ylabel('Max. Affinity Pressure')


def plot_strategy_boxplots(df, ax1=None, ax2=None):
    gdf = df[['instance', 'strategy', 'profit', 'pressure_max']].groupby(['instance', 'strategy'], as_index=False)
    pdf = gdf.agg({'profit': 'mean', 'pressure_max': 'mean'})
    pdf = pdf.sort_values(by=['profit', 'pressure_max'],
                          ascending=[False, True])  # .melt(id_vars=['instance', 'strategy'])
    pdf = pdf.rename(columns=FULL_LABELS)
    pdf = pdf.replace(to_replace=FULL_LABELS.keys(), value=FULL_LABELS.values())

    print(pdf)
    b1 = sns.boxplot(x='Strategy', y='Profit', ax=ax1, data=pdf)
    b2 = sns.boxplot(x='Strategy', y='Max. Affinity Pressure', ax=ax2, data=pdf)
    ax1.set_ylabel('$\sum Profit$')

    for item in b1.get_xticklabels() + b2.get_xticklabels():
        item.set_rotation(45)


def plot_ap_line(logdf, filename=None):
    _, ax = plt.subplots(figsize=figsize_column(1.0, height_ratio=0.6))
    cdf = logdf[['cycle', 'strategy', 'total_pressure_max']].groupby(['cycle', 'strategy'], as_index=False).mean()
    cdf = cdf.pivot(index='cycle', columns='strategy', values='total_pressure_max')
    labels = {**LABELS, **STRATEGY_SHORT}
    cdf = cdf.rename(columns=labels)
    # cdf = cdf.replace(to_replace=labels.keys(), value=labels.values())

    cdf.plot(ax=ax, legend=False, linewidth=1, sort_columns=True)
    ax.set_ylabel("Affinity Pressure")
    ax.set_xlabel("Cycle")
    # ax.set_ylim(bottom=0)
    ax.legend(ncol=3, loc=1, bbox_to_anchor=(0.84, 1.3), frameon=True, columnspacing=1.2)

    if filename:
        plt.savefig(filename, dpi=500, bbox_inches='tight', pad_inches=0)


def results_table(logdf, statdf, filename=None):
    agg_dict = {
        'used_agents': np.mean,
        'first_assigned': np.max,
        'max_cycles': np.max,
        'avg_cycles': np.mean,
        'distribution': np.mean,
        'rel_assignments': np.mean,
        'total_assignments': np.min,
    }

    grouped_logdf = logdf[['strategy', 'nb_agents', 'agentavail', 'taskavail', 'nb_tasks', 'profit']].groupby(
        ['strategy', 'nb_agents', 'nb_tasks', 'agentavail', 'taskavail'], as_index=False).sum()

    if 'mcmkp' in filename:
        log_file = 'mcmkp_60.p'
    elif 'tcsa' in filename:
        log_file = 'tcsa_60.p'
    elif 'mcmssp' in filename:
        log_file = 'mcmssp_60.p'
    else:
        log_file = None

    if log_file and os.path.isfile(log_file):
        rdf = pickle.load(open(log_file, 'rb'))['log']
        rdf = rdf[['strategy', 'nb_agents', 'agentavail', 'taskavail', 'nb_tasks', 'profit']].groupby(
            ['strategy', 'nb_agents', 'nb_tasks', 'agentavail', 'taskavail'], as_index=False).sum()

        # (rdf['strategy'] == 'profit') &
        grouped_logdf['max_profit'] = grouped_logdf[
            ['nb_agents', 'nb_tasks', 'agentavail', 'taskavail', 'strategy', 'profit']].apply(lambda x: rdf[
            (rdf['nb_agents'] == x['nb_agents']) & (rdf['nb_tasks'] == x['nb_tasks']) & (
                    rdf['agentavail'] == x['agentavail']) & (rdf['taskavail'] == x['taskavail'])]['profit'].max(),
                                                                                              axis=1)
        new_column = grouped_logdf.groupby(['nb_agents', 'nb_tasks', 'agentavail', 'taskavail'], as_index=False).apply(
            lambda x: x['profit'] / x['max_profit'])
        grouped_logdf['rel_profit'] = new_column.reset_index(level=0, drop=True)
    else:
        grouped_logdf['rel_profit'] = grouped_logdf.groupby(
            ['nb_agents', 'nb_tasks', 'agentavail', 'taskavail']).profit.transform(lambda x: x / x.max())

    del grouped_logdf['profit']

    tempdf = statdf[['strategy', 'nb_agents', 'nb_tasks', 'agentavail', 'taskavail', 'fullrotations']]
    tempdf = tempdf.groupby(['nb_agents', 'nb_tasks', 'agentavail', 'taskavail', 'strategy'], as_index=False)
    tempdf = tempdf.agg({'fullrotations': [np.min, np.mean]})

    grouped_statdf = statdf[['strategy', 'nb_agents', 'nb_tasks', 'agentavail', 'taskavail'] + list(agg_dict.keys())]
    grouped_statdf = grouped_statdf.groupby(['nb_agents', 'nb_tasks', 'agentavail', 'taskavail', 'strategy'],
                                            as_index=False).agg(agg_dict)
    grouped_statdf = grouped_statdf.merge(grouped_logdf)

    grouped_statdf[['fullrotations', 'avgrotations']] = tempdf['fullrotations']

    result_cols = ['nb_agents', 'nb_tasks', 'agentavail', 'taskavail', 'strategy', 'rel_profit', 'first_assigned',
                   'fullrotations', 'avgrotations']

    # First group by tasks/agents + availabilities
    grouped_size_avail = grouped_statdf[result_cols]
    grouped_size_avail = grouped_size_avail.groupby(['nb_agents', 'nb_tasks', 'agentavail', 'taskavail', 'strategy'],
                                                    as_index=False).mean()
    grouped_size_avail['rel_profit'] *= 100
    grouped_size_avail['agentavail'] = (grouped_size_avail['agentavail'] * 100).astype(int)
    grouped_size_avail['taskavail'] = (grouped_size_avail['taskavail'] * 100).astype(int)
    grouped_size_avail = grouped_size_avail.pivot_table(index=['nb_agents', 'nb_tasks', 'agentavail', 'taskavail'],
                                                        columns='strategy', margins=True)

    grouped_size_avail['first_assigned'] = grouped_size_avail['first_assigned'].astype(int, errors='ignore')
    grouped_size_avail['fullrotations'] = grouped_size_avail['fullrotations'].astype(int, errors='ignore')

    # Group again only by tasks/agents
    grouped_size = grouped_size_avail.groupby(['nb_agents', 'nb_tasks']).mean()
    grouped_avail = grouped_size_avail.groupby(['agentavail', 'taskavail']).mean()

    def rotation_string(df):
        df['fullrotations'] = df['fullrotations'].apply(lambda x: (np.floor(x * 10) / 10).map(str))
        df['avgrotations'] = df['avgrotations'].apply(
            lambda x: "(" + (np.floor(x * 10) / 10).map(str) + ")")
        df['fullrotations'] = df['fullrotations'] + " " + df['avgrotations']
        return df

    grouped_size = rotation_string(grouped_size)
    grouped_avail = rotation_string(grouped_avail)
    grouped_size_avail = rotation_string(grouped_size_avail)

    del grouped_size['avgrotations']
    del grouped_size_avail['avgrotations']

    # Export tables
    # Size
    grouped_size = grouped_size.rename(columns={**LABELS, **STRATEGY_SHORT})
    grouped_size_filename = 'size_' + filename if filename else None
    export_tables(grouped_size, grouped_size_filename)

    # Availability
    grouped_avail = grouped_avail.rename(columns={**LABELS, **STRATEGY_SHORT})
    grouped_avail_filename = 'avail_' + filename if filename else None
    export_tables(grouped_avail, grouped_avail_filename)

    # Size + Availability
    grouped_size_avail = grouped_size_avail.rename(columns={**LABELS, **STRATEGY_SHORT})
    grouped_size_avail_filename = 'avail_size_' + filename if filename else None
    export_tables(grouped_size_avail, grouped_size_avail_filename)

    return grouped_size_avail


def export_tables(df, filename=None):
    if filename and os.path.isfile(filename):
        os.unlink(filename)

    df = df.round(1)

    profit_table = df[[LABELS['rel_profit']]]  # .drop('FOP', axis=1, level=1)
    output_table = df[[LABELS['fullrotations']]].join(profit_table)
    print(df_to_latex(output_table.transpose(), filename))


def df_to_latex(df, filename=None):
    table = df.to_latex(multicolumn_format='l')
    table = table.replace('nb\_agents', 'Avail. Agents')
    table = table.replace('nb\_tasks', 'Avail. Tasks')
    table = table.replace('strategy', '')
    table = table.replace('\\{\\}', '')

    if filename:
        open(filename, 'a').write('\n' + table)

    return table


def plot_pareto_mcmkp():
    mcmkp = pd.DataFrame([
        ('switch10', 2.50, 86.2, -0.5, -0.3),
        ('switch20', 2.43, 88.4, 0.05, -0.3),
        ('switch30', 1.37, 91.0, -0.5, -0.3),
        ('switch40', 1.32, 93.0, 0.05, 0.1),
        ('productcomb', 0.50, 91.0, -0.3, 0.1),
        ('wppind', 1.46, 88.3, -0.5, 0.1),
        ('affinity', 2.53, 85.1, -0.5, 0.1),
        ('profit', 0.17, 100., -1.15, -0.0)
    ], columns=('Strategy', 'Rotations', 'Profit', 'POff', 'ROff'))
    mcmkp_limit = pd.DataFrame([
        ('switch10', 2.51, 86.0, -0.3, 0.1),
        ('switch20', 2.48, 87.6, -0.3, 0.1),
        ('switch30', 1.44, 89.6, -0.5, 0.1),
        ('switch40', 1.41, 91.0, -0.4, 0.1),
        ('productcomb', 0.50, 90.8, -0.5, -0.3),
        ('wppind', 3.52, 84.1, -0.5, -0.3),
        ('affinity', 2.53, 85.1, -0.5, -0.3),
        ('profit', 0.33, 94.3, -1.15, -0.0)
    ], columns=('Strategy', 'Rotations', 'Profit', 'POff', 'ROff'))
    name = 'mcmkp_pareto.pgf'
    plot_pareto(mcmkp, mcmkp_limit, name)


def plot_pareto_tcsa():
    without = pd.DataFrame([
        ('switch10', 6.138, 79.6, 0.05, -0.55),
        ('switch20', 5.128, 80.5, -2.8, -0.2),
        ('switch30', 4.121, 82.2, -1, -0.6),
        ('switch40', 4.116, 84.1, 0.3, -0.1),
        ('productcomb', 7.135, 96.3, -0.4, 0.25),
        ('wppind', 6.131, 74.2, -0.9, 0.2),
        ('affinity', 6.145, 79.2, -1.6, 0.15),
        ('profit', 1.096, 100., -1.9, 0.0)
    ], columns=('Strategy', 'Rotations', 'Profit', 'POff', 'ROff'))
    withlimit = pd.DataFrame([
        ('switch10', 6.139, 79.8, 0.05, 0.2),
        ('switch20', 5.134, 80.9, -0.7, -0.6),
        ('switch30', 5.131, 83.0, -0.7, 0.2),
        ('switch40', 5.129, 85.3, 0.35, -0.1),
        ('productcomb', 6.135, 96.3, -0.4, 0.25),
        ('wppind', 8.132, 72.2, -0.8, -0.7),
        ('affinity', 6.145, 79.2, -1.6, 0.15),
        ('profit', 2.126, 100., -1.9, 0.05)
    ], columns=('Strategy', 'Rotations', 'Profit', 'POff', 'ROff'))
    name = 'tcsa_pareto.pgf'
    plot_pareto(without, withlimit, name)


def plot_pareto(without, withlimit, filename):
    _, ax = plt.subplots(figsize=figsize_text(1.0, height_ratio=0.8))
    without.plot.scatter(x='Profit', y='Rotations', c='r', ax=ax, s=40, marker='.')
    withlimit.plot.scatter(x='Profit', y='Rotations', c='k', ax=ax, s=40, marker='x')

    for s, p, r, poff, roff in zip(without['Strategy'], without['Profit'], without['Rotations'], without['POff'],
                                   without['ROff']):
        ax.annotate(STRATEGY_SHORT[s], (p + poff, r + roff))

    for s, p, r, poff, roff in zip(withlimit['Strategy'], withlimit['Profit'], withlimit['Rotations'],
                                   withlimit['POff'], withlimit['ROff']):
        ax.annotate(STRATEGY_SHORT[s], (p + poff, r + roff))

    pure_patch = mlines.Line2D([], [], marker='.', color='r', linewidth=0, markersize=7, label='Pure Strategy')
    la_patch = mlines.Line2D([], [], marker='x', color='k', linewidth=0, markersize=7, label='With Limited Assignment')
    plt.legend(handles=[pure_patch, la_patch])
    ax.set_xlabel('Relative Profit (\% of FOP)')
    ax.set_ylabel('Rotations')
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['show', 'save', 'paper'])
    parser.add_argument('files', nargs='+')
    parser.add_argument('-n', '--name', default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    if not os.path.isfile(args.name + '.p'):
        results = []

        for f in args.files:
            if f.endswith('_log.csv'):
                try:
                    results.append(read_results(f))
                except Exception as e:
                    print(f, e)

        log_dfs, assign_dfs, stats_dfs = zip(*results)
        logdf = pd.concat(log_dfs, sort=True)
        assign_df = pd.concat(assign_dfs, sort=True)
        statdf = pd.concat(stats_dfs, sort=True)

        writer = pd.ExcelWriter(args.name + '.xlsx')
        statdf.to_excel(writer, 'Stats')
        logdf.to_excel(writer, 'Log')
        # assign_df.to_excel(writer, 'Assignment')
        writer.save()

        pickle.dump({
            'log': logdf,
            'assign': assign_df,
            'stat': statdf
        }, open(args.name + '.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        x = pickle.load(open(args.name + '.p', 'rb'))
        logdf = x['log']
        assign_df = x['assign']
        statdf = x['stat']

    if args.action == 'paper':
        # plot_profit_bars(logdf, args.name + '_profit.pgf')
        # sns.set_palette('colorblind', color_codes=True)
        # plot_ap_line(logdf, args.name + '_ap.pgf')

        # plot_rel_profit_boxplots(logdf, args.name + '_profit.pgf')

        results_table(logdf, statdf, args.name + '.tex')

        sys.exit(0)

    plt.figure(figsize=(20, 10))

    if len(logdf.instance.unique()) == 0:
        print('No full solution set')
        sys.exit(1)
    elif len(logdf.instance.unique()) == 1:

        ax = plt.subplot(3, 3, 1)
        plot_timeline(logdf, ['pressure_max', 'pressure_mean'], ax=ax)
        ax.set_title('Affinity Pressure')

        ax = plt.subplot(3, 3, 2)
        plot_strategy_front(logdf, ax=ax)

        ax = plt.subplot(3, 3, 3)
        plot_timeout_bars(logdf, ax=ax)

        ax = plt.subplot(3, 3, 4)
        plot_bars(statdf, 'used_agents', ax=ax)
        ax.set_title('Used Agents (per Task) (Higher)')

        ax = plt.subplot(3, 3, 5)
        plot_bars(statdf, 'first_assigned', agg=max, ax=ax)
        ax.set_title('Time to First Assignment (Lower)')

        ax = plt.subplot(3, 3, 6)
        plot_bars(statdf, 'avg_cycles', ax=ax)
        ax.set_title('Assignment Frequency (Closer to 1)')

        ax = plt.subplot(3, 2, 5)
        plot_timeline(logdf, ['profit'], ax=ax)
        ax.set_title('Assigned Profit (Higher)')

        ax = plt.subplot(3, 2, 6)
        plot_timeline(logdf, ['affinity'], ax=ax)
    else:
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        plot_strategy_boxplots(logdf, ax1=ax1, ax2=ax2)
        ax1.set_title('Assigned Profit (Higher)')
        ax2.set_title('Max. Affinity Pressure (Lower)')

        ax = plt.subplot(2, 3, 3)
        plot_boxplot(statdf, 'avg_cycles', ax=ax)
        ax.set_title('Assignment Frequency (Closer to 1)')

        ax = plt.subplot(2, 4, 5)
        plot_boxplot(statdf, 'used_agents', ax=ax)
        ax.set_title('Used Agents (per Task) (Higher)')

        ax = plt.subplot(2, 4, 6)
        plot_boxplot(statdf, 'used_tasks', ax=ax)
        ax.set_title('Assigned Tasks (Higher)')

        ax = plt.subplot(2, 4, 7)
        plot_bars(statdf, 'first_assigned', agg=np.max, ax=ax)
        ax.set_title('Time to First Assignment (Lower)')

        ax = plt.subplot(2, 4, 8)
        plot_boxplot(statdf, 'distribution', ax=ax)
        ax.set_title('Distribution between Task and Agents (MAD) (Lower)')

    if args.action == 'show':
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()
    else:
        filename = args.name
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pgf', dpi=300, bbox_inches='tight')
