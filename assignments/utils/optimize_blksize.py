#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


time_token = 'time ='

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
def collect_seq_data(program, sizes, BLK):
    times = []
    problem_size_means = []
    for problem_size in sizes:
        print("Problem size:", problem_size)
        for blksize in BLK:
            print("Block size:", blksize)
            results = []
            for i in range(0, 8): 
                output = subprocess.run([program['program_name']] + [str(problem_size), str(blksize)], capture_output=True).stdout.decode('utf-8')
                print("result: ", output)
                result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
                print(result)
                results.append(float(result))
            results.remove(max(results))
            results.remove(min(results))
            mean = sum(results)/len(results)
            problem_size_means.append((problem_size, blksize, mean))
    return problem_size_means 


# index represents the index of sizes array
def calculate_speedup(baseline_mean, test_times):
    speedup_list = []
    # datum = (threads, problem_size, mean_time)
    for datum in test_times:
        speedup = baseline_mean / datum[2] * 100
        speedup_list.append((datum[0], datum[1], speedup))
    return speedup_list

BLK = [10, 100, 1000, 10000, 100000]
#sizes = [500000 , 1000000, 1500000]
sizes = [500000000, 1000000000, 1500000000] #500mil, 1bil, 1.5bil

#sieve1_results = collect_seq_data(sieve1, sizes)
#baseline_mean = sum(sieve1_results)/len(sieve1_results)

#sieve2_results = collect_data(sieve2, sizes)
#sieve2_speedups = calculate_speedup(baseline_mean, sieve2_results)

sieve3_results = collect_seq_data(sieve3, sizes, BLK)


print(f"len(BLK) = {len(BLK)}, data: {len([datum[2] for datum in sieve3_results if datum[0] == sizes[0]])}")
data = pd.DataFrame({
    "BLKSIZE": BLK,
    "500M": [datum[2] for datum in sieve3_results if datum[0] == sizes[0]], 
    #"500K 100K": [datum[2] for datum in sieve3_results if datum[0] == sizes[0]], 
    #"500K 500K": [datum[2] for datum in sieve3_results if datum[0] == sizes[0]],
    #"500K 2000K": [datum[2] for datum in sieve3_results if datum[0] == sizes[0]], 
    "1000M": [datum[2] for datum in sieve3_results if datum[0] == sizes[1]],
    #"1000K 100K": [datum[2] for datum in sieve3_results if datum[0] == sizes[1]],
    #"1000K 500K": [datum[2] for datum in sieve3_results if datum[0] == sizes[1]],
    #"1000K 1000K": [datum[2] for datum in sieve3_results if datum[0] == sizes[1] and datum[1] == BLK[3]],
    #"1000K 2000K": [datum[2] for datum in sieve3_results if datum[0] == sizes[1] and datum[1] == BLK[4]],
    "1500M": [datum[2] for datum in sieve3_results if datum[0] == sizes[2]]
    #"1500K 100K": [datum[2] for datum in sieve3_results if datum[0] == sizes[2]],
    #"1500K 500K": [datum[2] for datum in sieve3_results if datum[0] == sizes[2]]
    #"1500K 1000K": [datum[2] for datum in sieve3_results if datum[0] == sizes[2] and datum[1] == BLK[3]],
    #"1500K 2000K": [datum[2] for datum in sieve3_results if datum[0] == sizes[2] and datum[1] == BLK[4]]

})


filename = 'sieve3'
data.to_csv(filename + '.csv', index = False)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve3 500M Statistics')
ax.scatter(data['BLKSIZE'], data['500M'], label = '500M')
plt.xlabel('BLKSIZE')
plt.ylabel('Execution Time (s)')
#plt.legend()
plt.savefig('sieve3-500M.png')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve3i 1000M Statistics')
ax.scatter(data['BLKSIZE'], data['1000M'], label = '1000M')
plt.xlabel('BLKSIZE')
plt.ylabel('Execution Time (s)')
#plt.legend()
plt.savefig('sieve3-1000M.png')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve3 1500M Statistics')
ax.scatter(data['BLKSIZE'], data['1500M'], label = '1500M')
plt.xlabel('BLKSIZE')
plt.ylabel('Execution Time (s)')
#plt.legend()
plt.savefig('sieve3-1500M.png')
#plt.show()    

