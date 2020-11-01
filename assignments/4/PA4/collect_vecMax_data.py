#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


time_token = 'Compute time:'

vecMax00 = {
    'program_name': './vecMax00',
}

vecMax01 = {
    'program_name': './vecMax01',
}

# [program] Dict
def collect_seq_data(program):
    results = []
    for i in range(0, 8): 
        output = subprocess.run([program['program_name']] + str("80 128 1280000000").split(), capture_output=True).stdout.decode('utf-8')
        result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
        print(result)
        results.append(float(result))
    results.remove(max(results))
    results.remove(min(results))
    mean = sum(results)/len(results)

    return mean 


# index represents the index of sizes array
def calculate_speedup(baseline_mean, test_time):
    speedup_list = []
    # datum = (threads, problem_size, mean_time)
    speedup = baseline_mean / test_time * 100
    return speedup


vecMax00_mean = collect_seq_data(vecMax00)

vecMax01_mean = collect_seq_data(vecMax01)
speedup = calculate_speedup(vecMax00_mean, vecMax01_mean)

print(f"00 mean: {vecMax00_mean}, 01 mean: {vecMax01_mean}, speedup: {speedup}")

"""
data = pd.DataFrame({
    "Threads": range(1,9),
    "0.5B": [datum[2] for datum in sieve3_speedups if datum[1] == sizes[0]],
    "1.0B": [datum[2] for datum in sieve3_speedups if datum[1] == sizes[1]],
    "1.5B": [datum[2] for datum in sieve3_speedups if datum[1] == sizes[2]]
})


filename = 'sieve4'
data.to_csv(filename + '.csv', index = False)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve4 vs sieve3 Speedup 0.5B')
ax.plot(data['Threads'], data['0.5B'], label = 'sieve4')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.savefig('sieve4-speedup1.png')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve4 vs sieve3 Speedup 1B')
ax.plot(data['Threads'], data['1.0B'], label = 'sieve4')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.savefig('sieve4-speedup2.png')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('sieve4 vs sieve3 Speedup 1.5B')
ax.plot(data['Threads'], data['1.5B'], label = 'sieve4')
plt.xlabel('Threads')
plt.ylabel('Speedup (%)')
plt.savefig('sieve4-speedup3.png')


#plt.show()    
"""
