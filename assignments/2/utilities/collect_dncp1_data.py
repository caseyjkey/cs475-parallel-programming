#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

program_name = './MMScanDNCP1'
arguments = '500000 6'
time_token = 'DNC:'

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
    ax.set_xlabel("Tasks")
    return ax


mean_times = [] # [[threads, mean], ...]
for p in [1, 10, 100, 1000, 10000, 40000, 60000, 100000]:
    print("Threads: 16")
    os.environ['OMP_NUM_THREADS'] = str(16)
    results = []
    for i in range(0, 8): 
        output = subprocess.run([program_name] + (arguments + ' ' + str(p)).split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
        print(result)
        results.append(float(result))
    results.remove(max(results))
    results.remove(min(results))
    mean = sum(results)/len(results)
    print('Mean:', mean, '\n')
    mean_times.append((p, mean))

# Create efficiency and speedup dataframes
# data = [[threads, mean, speedup, efficiency], ...]
seq_data = [5.782719, 5.944667, 5.933426, 5.946885, 5.946885, 5.933235, 5.946597]
seq_data.remove(max(seq_data))
seq_data.remove(min(seq_data))
seq_mean = sum(seq_data)/len(seq_data)

speedup_list = []
efficiency_list = []
for i, datum in enumerate(mean_times):
    p = datum[0]
    mean = datum[1]
    speedup = seq_mean / mean 
    efficiency = speedup / p * 100
    speedup_list.append((p, speedup))
    efficiency_list.append((p, efficiency))

data = pd.DataFrame({
    "Tasks": [1, 10, 100, 1000, 10000, 100000],
    "Mean Execution Time (s)": [datum[1] for datum in mean_times]
#    "Speedup (%)": [datum[1] for datum in speedup_list],
#    "Efficiency (%)": [datum[1] for datum in efficiency_list]
})

filename = program_name.split('/')[1] 
data.to_csv(filename + '.csv', index = False)
data.plot(x='Tasks', y='Mean Execution Time (s)', kind = 'scatter')
#plot_multi(data, figsize=(10, 5))
#plt.xticks(np.arange(8), np.arange(1, 9))
#plt.subplots_adjust(right=0.8)
plt.title(filename + ' Statistics')
plt.savefig(filename + '.png')
plt.show()    

