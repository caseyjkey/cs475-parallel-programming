#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

baseline = {
    'program_name': './syr2k',
    'arguments': '1500 1500 2>/dev/null',
    'time_token': 'time='
}

perm = {
    'program_name': './syr2k-perm',
    'arguments': '1500 1500 2>/dev/null',
    'time_token': 'time='
}

baselineP = {
    'program_name': './syr2kP',
    'arguments': '1500 1500 2>/dev/null',
    'time_token': 'time='
}

permP = {
    'program_name': './syr2k-permP',
    'arguments': '1500 1500 2>/dev/null',
    'time_token': 'time='
}

def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = ['red', 'green', 'blue']

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])
        print(cols[n], 'max', data[cols[n]].max())
        print(cols[n], 'data', data[cols[n]])
        ax_new.set_ylim([0, data.loc[:, cols[n]].max()])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    ax.set_xlabel("Iteration")
    return ax

# [program] Dict
def collect_data(program):
    times = []
    thread_means = []
    for threads in range(1, 11):
        print("Threads:", threads)
        os.environ['OMP_NUM_THREADS'] = str(threads)

        results = []
        for i in range(0, 8): 
            output = subprocess.run([program['program_name']] + program['arguments'].split(), capture_output=True).stdout.decode('utf-8')
            result = re.search('(?<=' + program['time_token'] + ').\S*', output).group(0).strip()
            print(result)
            results.append(float(result))
        results.remove(max(results))
        results.remove(min(results))
        mean = sum(results)/len(results)
        thread_means.append(mean)
    return thread_means 

def calculate_speedup(baseline_mean, test_times):
    speedup_list = []
    for time in test_times:
        speedup = baseline_mean / time * 100
        speedup_list.append(speedup)
    return speedup_list


baseline_results = collect_data(baseline)
baseline_mean = sum(baseline_results)/len(baseline_results)

perm_results = collect_data(perm)

baselineP_results = collect_data(baselineP)

permP_results = collect_data(permP)

perm_speedups = calculate_speedup(baseline_mean, perm_results)
baselineP_speedups = calculate_speedup(baseline_mean, baselineP_results)
permP_speedups = calculate_speedup(baseline_mean, permP_results)

data = pd.DataFrame({
    "Threads": [i for i in range(1, len(perm_speedups)+1)],
    "Permuted Speedup (%)": perm_speedups,
    "Baseline Parallelized Speedup (%)": baselineP_speedups,
    "Permuted Parallelized Speedup (%)": permP_speedups
})

filename = 'Locality'
data.to_csv(filename + '.csv', index = False)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(data['Threads'], data['Permuted Speedup (%)'], label = 'Permuted')
ax.plot(data['Threads'], data['Baseline Parallelized Speedup (%)'], label = 'Baseline Parallelized')
ax.plot(data['Threads'], data['Permuted Parallelized Speedup (%)'], label = 'Permuted Parallelized')

plt.xlabel('Threads')
plt.ylabel('Speedup (%)')

#plot_multi(data, figsize=(10, 5))
#plt.xticks(np.arange(8), np.arange(1, 9))
#plt.subplots_adjust(right=0.8)
plt.title('Locality' + ' Statistics')
plt.legend()
plt.savefig('Locality' + '.png')
plt.show()    

