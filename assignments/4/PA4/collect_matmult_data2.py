#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


time_token = 'Time: '

#matmult00 = ['./matmult00', './matmult10', './matmult20', './matmult30']
matmult00 = ['./matmult00' for i in range(0,4)]

matmult01 = ['./matmult01', './matmult11', './matmult21', './matmult31']

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
                result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
                print((threads, size, result))
                results.append(float(result))
            results.remove(max(results))
            results.remove(min(results))
            mean = sum(results)/len(results)
            thread_means.append((threads, size, mean))
    return thread_means 

# [program] Dict
def collect_seq_data(programs, sizes):
    times = []
    means = []
    for j, program in enumerate(programs):
        print("Program:", program, sizes[j])

        results = []
        for i in range(0, 8): 
            output = subprocess.run([program] + str(sizes[j]).split(), capture_output=True).stdout.decode('utf-8')
            print(output)
            result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
            print(result)
            results.append(float(result))
        results.remove(max(results))
        results.remove(min(results))
        mean = sum(results)/len(results)
        means.append(mean)
    return means 


# index represents the index of sizes array
def calculate_speedup(baseline_mean, test_times):
    speedup_list = []
    # datum = (threads, problem_size, mean_time)
    for datum in test_times:
        print("datum", datum)
        speedup = baseline_mean / datum * 100
        speedup_list.append(speedup)
    return speedup_list


sizes = [100, 200, 400, 800]
#sizes = [160, 320, 640]

matmult00_results = collect_seq_data(matmult00, [100, 100, 100, 100]) #sizes)
baseline_mean = sum(matmult00_results)/len(matmult00_results)

matmult01_results = collect_seq_data(matmult01, [50, 50, 50, 50])
speedup = calculate_speedup(baseline_mean, matmult01_results)

data = pd.DataFrame({
    "BLOCK_SIZE": [16, 8, 4, 2],
    "matmult00": matmult00_results,
    "matmult01": matmult01_results
})


filename = 'matmult-blocks'
data.to_csv(filename + '.csv', index = False)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('matmult00 vs matmult01 - BLOCK_SIZE')
ax.scatter(data['BLOCK_SIZE'], data['matmult00'], label = 'matmult00')
ax.scatter(data['BLOCK_SIZE'], data['matmult01'], label = 'matmult01')
plt.xlabel('BLOCK_SIZE')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.savefig(filename + '-1.png')

"""
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve4 vs sieve3 Speedup 1.5B')
ax.plot(data['Threads'], data['1.5B'], label = 'sieve4')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.savefig('sieve4-speedup3.png')
"""

#plt.show()    

