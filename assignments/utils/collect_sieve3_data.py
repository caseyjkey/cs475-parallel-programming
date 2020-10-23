#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


time_token = 'time ='

sieve1 = {
    'program_name': './sieve1',
}

sieve3 = {
    'program_name': './sieve3',
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
            result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
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
        print("datum", datum)
        speedup = baseline_mean / datum * 100
        speedup_list.append(speedup)
    return speedup_list


#sizes = [500000 , 1000000, 1500000]
sizes = [500000000, 1000000000, 1500000000] #500mil, 1bil, 1.5bil

sieve1_results = collect_seq_data(sieve1, sizes)
baseline_mean = sum(sieve1_results)/len(sieve1_results)

sieve3_results = collect_seq_data(sieve3, sizes)
sieve3_speedups = calculate_speedup(baseline_mean, sieve3_results)

data = pd.DataFrame({
    "Problem Size": sizes,
    "sieve3": sieve3_speedups

})


filename = 'sieve3'
data.to_csv(filename + '.csv', index = False)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve3 vs sieve1 Speedup')
ax.plot(data['Problem Size'], data['sieve3'], label = 'sieve3')
plt.xlabel('Problemsize')
plt.ylabel('Speedup (%)')
plt.legend()
plt.savefig('sieve3-speedup.png')

#plt.show()    

