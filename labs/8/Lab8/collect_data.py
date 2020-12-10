#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

seq = {
    'program_name': './prog_SEQ',
    'time_token': 'time ='
}

omp = {
    'program_name': './prog_OMP',
    'time_token': 'time ='
}

def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = ['red', 'green', 'blue']

    # First axis
    ax = data.loc[:, cols[0]].plot(kind='scatter', label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(kind='scatter', ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])
        ax_new.set_ylim([0, data.loc[:, cols[n]].max()])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    ax.set_xlabel("Iteration")
    return ax

# [program] Dict
def collect_data(program, sizes):
    times = []
    thread_means = []
    for threads in range(1, 9):
        print("Threads:", threads)
        os.environ['OMP_NUM_THREADS'] = str(threads)
        for size in sizes:
            results = []
            for i in range(0, 8): 
                output = subprocess.run([program['program_name']] + str(size).split(), capture_output=True).stdout.decode('utf-8')
                result = re.search('(?<=' + program['time_token'] + ').\S*', output).group(0).strip()
                print((threads, size, result))
                results.append(float(result))
            results.remove(max(results))
            results.remove(min(results))
            mean = sum(results)/len(results)
            thread_means.append((threads, size, mean))
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


# index represents the index of sizes array
def calculate_speedup(baseline_mean, test_times):
    speedup_list = []
    # datum = (threads, problem_size, mean_time)
    for datum in test_times:
        speedup = baseline_mean / datum[2] * 100
        speedup_list.append((datum[0], datum[1], speedup))
    return speedup_list


sizes = [300, 3000, 30000]

seq_results = collect_seq_data(seq, sizes)
baseline_mean = sum(seq_results)/len(seq_results)

omp_results = collect_data(omp, sizes)
omp_speedups = calculate_speedup(baseline_mean, omp_results)


print(omp_speedups[0])
data = pd.DataFrame({
    "Threads": [i for i in range(1, 9)],
    "300": [speedup[2] for speedup in omp_speedups if speedup[1] == sizes[0]], 
    "3K": [speedup[2] for speedup in omp_speedups if speedup[1] == sizes[1]],
    "30K": [speedup[2] for speedup in omp_speedups if speedup[1] == sizes[2]]
})


filename = 'test'
data.to_csv(filename + '.csv', index = False)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Speedup SEQ vs OMP')
ax.scatter(data['Threads'], data['300'], label = '300')
ax.scatter(data['Threads'], data['3K'], label = '3K')
ax.scatter(data['Threads'], data['30K'], label = '30K')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.legend()
plt.savefig('results.png')

plt.show()    

