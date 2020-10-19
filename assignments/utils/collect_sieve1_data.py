#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sieve = {
    'program_name': './sieve',
    'arguments': '100',
    'time_token': 'time ='
}

sieve1 = {
    'program_name': './sieve1',
    'arguments': '100',
    'time_token': 'time ='
}

sieve2 = {
    'program_name': './sieve2-1',
    'arguments': '100',
    'time_token': 'time ='
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
    for threads in range(1, 9):
        print("Threads:", threads)
        os.environ['OMP_NUM_THREADS'] = str(threads)
        for size in [500000000, 1000000000, 1500000000]:
            results = []
            for i in range(0, 8): 
                output = subprocess.run([program['program_name']] + str(size).split(), capture_output=True).stdout.decode('utf-8')
                result = re.search('(?<=' + program['time_token'] + ').\S*', output).group(0).strip()
                print(threads, size, result)
                results.append(float(result))
            results.remove(max(results))
            results.remove(min(results))
            mean = sum(results)/len(results)
            thread_means.append(threads, size, mean)
    return thread_means 

# [program] Dict
def collect_seq_data(program, sizes):
    times = []
    problem_size_means = []
    for problem_size in sizes:
        print("Problem size:", problem_size)

        results = []
        for i in range(0, 8): 
            output = subprocess.run([program['program_name']] + str(problem_size).split(), capture_output=True).stdout.decode('utf-8')
            result = re.search('(?<=' + program['time_token'] + ').\S*', output).group(0).strip()
            print(result)
            results.append(float(result))
        results.remove(max(results))
        results.remove(min(results))
        mean = sum(results)/len(results)
        problem_size_means.append(mean)
    return problem_size_means 


def calculate_speedup(baseline_mean, test_times):
    speedup_list = []
    for time in test_times:
        speedup = baseline_mean / time * 100
        speedup_list.append(speedup)
    return speedup_list


# Uncomment this to compare sieve and sieve1
#baseline_results = collect_seq_data(sieve, [100000, 200000, 300000])
#sieve1_results = collect_seq_data(sieve1, [100000, 200000, 300000])


sieve1_results = collect_seq_data(sieve1, [500000000, 1000000000, 1500000000])
baseline_mean = sum(sieve1_results)/len(sieve1_results)

sieve2_results = collect_data(sieve2)
sieve2_speedups = calculate_speedup(baseline_mean, sieve2_results)

seq_data = pd.DataFrame({
    "Problem Size": [100000, 200000, 300000],
    "Sieve": baseline_results,
    "Sieve1": sieve1_results
})


data = pd.DataFrame({
    "Threads": [i for i in range(1, len(perm_speedups)+1)],
    "Sieve1 Speedup (%)": perm_speedups,
    "Baseline Parallelized Speedup (%)": baselineP_speedups,
    "Permuted Parallelized Speedup (%)": permP_speedups
})


filename = 'sieve'
seq_data.to_csv(filename + '.csv', index = False)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(seq_data['Problem Size'], seq_data['Sieve'], label = 'Sieve')
ax.plot(seq_data['Problem Size'], seq_data['Sieve1'], label = 'Sieve1')

plt.xlabel('Problem Size')
plt.ylabel('Execution Time (s)')

#plot_multi(data, figsize=(10, 5))
#plt.xticks(np.arange(8), np.arange(1, 9))
#plt.subplots_adjust(right=0.8)
plt.title('SeqSieve' + ' Statistics')
plt.legend()
plt.savefig('Sieve' + '.png')
plt.show()    

