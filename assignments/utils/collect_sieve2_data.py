#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sieve1 = {
    'program_name': './sieve1',
    'arguments': '100',
    'time_token': 'time ='
}

sieve3 = {
    'program_name': './sieve3-1',
    'arguments': '100',
    'time_token': 'time ='
}

sieve3_1 = {
    'program_name': './sieve3-2',
    'arguments': '100',
    'time_token': 'time ='
}

sieve3_2 = {
    'program_name': './sieve3-3',
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


#sizes = [500000 , 1000000, 1500000]
sizes = [500000000, 1000000000, 1500000000] #500mil, 1bil, 1.5bil

sieve1_results = collect_seq_data(sieve1, sizes)
baseline_mean = sum(sieve1_results)/len(sieve1_results)

sieve3_results = collect_data(sieve3, sizes)
sieve3_speedups = calculate_speedup(baseline_mean, sieve3_results)

sieve3_1_results = collect_data(sieve3_1, sizes)
sieve3_1_speedups = calculate_speedup(baseline_mean, sieve3_1_results)

sieve3_2_results = collect_data(sieve3_2, sizes)
sieve3_2_speedups = calculate_speedup(baseline_mean, sieve3_2_results)


print(sieve3_speedups[0])
data = pd.DataFrame({
    "Threads": [i for i in range(1, 9)],
    "0.5B 3-1": [speedup[2] for speedup in sieve3_speedups if speedup[1] == sizes[0]], 
    "1.0B 3-1": [speedup[2] for speedup in sieve3_speedups if speedup[1] == sizes[1]],
    "1.5B 3-1": [speedup[2] for speedup in sieve3_speedups if speedup[1] == sizes[2]],
    "0.5B 3-2": [speedup[2] for speedup in sieve3_1_speedups if speedup[1] == sizes[0]], 
    "1.0B 3-2": [speedup[2] for speedup in sieve3_1_speedups if speedup[1] == sizes[1]],
    "1.5B 3-2": [speedup[2] for speedup in sieve3_1_speedups if speedup[1] == sizes[2]],
    "0.5B 3-3": [speedup[2] for speedup in sieve3_2_speedups if speedup[1] == sizes[0]],
    "1.0B 3-3": [speedup[2] for speedup in sieve3_2_speedups if speedup[1] == sizes[1]],
    "1.5B 3-3": [speedup[2] for speedup in sieve3_2_speedups if speedup[1] == sizes[2]]

})


filename = 'sieve3'
data.to_csv(filename + '.csv', index = False)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve3 0.5B Statistics')
ax.scatter(data['Threads'], data['0.5B 2-1'], label = '2-1')
ax.scatter(data['Threads'], data['0.5B 2-2'], label = '2-2')
ax.scatter(data['Threads'], data['0.5B 2-3'], label = '2-3')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.legend()
plt.savefig('sieve3-test1.png')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve3 1.0B Statistics')
ax.scatter(data['Threads'], data['1.0B 2-1'], label = '2-1')
ax.scatter(data['Threads'], data['1.0B 2-2'], label = '2-2')
ax.scatter(data['Threads'], data['1.0B 2-3'], label = '2-3')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.legend()
plt.savefig('sieve3-test2.png')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve3 1.5B Statistics')
ax.scatter(data['Threads'], data['1.5B 2-1'], label = '2-1')
ax.scatter(data['Threads'], data['1.5B 2-2'], label = '2-2')
ax.scatter(data['Threads'], data['1.5B 2-3'], label = '2-3')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.legend()
plt.savefig('sieve3-test3.png')

#plt.show()    

