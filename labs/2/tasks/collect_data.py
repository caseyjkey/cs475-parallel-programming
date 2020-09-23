#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

program_name = './Merge_sortP'
arguments = '1000000'
time_token = 'time ='

def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = ['red', 'green', 'blue']

    # First axis
    print(data.loc[:, cols[0]])
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])
        ax_new.set_ylim([0, data.loc[:, cols[n]].max() + 10])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    ax.set_xlabel("Threads")
    return ax


mean_times = [] # [[threads, mean], ...]
for threads in range (1, 9):
    print("Threads:", threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    results = []
    for i in range(0, 8): 
        output = subprocess.run([program_name] + arguments.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
        print(result)
        results.append(float(result))
    results.remove(max(results))
    results.remove(min(results))
    mean = sum(results)/len(results)
    print('Mean:', mean, '\n')
    mean_times.append((threads, mean))

# Create efficiency and speedup dataframes
# data = [[threads, mean, speedup, efficiency], ...]
seq_data = [19.538881, 19.508363, 19.510208, 19.525412, 19.512387, 19.512093, 19.511672]
seq_data.remove(max(seq_data))
seq_data.remove(min(seq_data))
seq_mean = sum(seq_data)/len(seq_data)

speedup_list = []
efficiency_list = []
for i, datum in enumerate(mean_times):
    threads = datum[0]
    mean = datum[1]
    speedup = seq_mean / mean
    efficiency = speedup / threads
    speedup_list.append((threads, speedup))
    efficiency_list.append((threads, efficiency))

print("Eff", efficiency_list)
print("Speedup", speedup_list)

data = pd.DataFrame({
    "Mean Execution Time (s)": [datum[1] for datum in mean_times],
    "Speedup (%)": [datum[1] for datum in speedup_list],
    "Efficiency (%)": [datum[1] for datum in efficiency_list]
})

filename = program_name.split('/')[1] 
data.to_csv(filename + '.csv', index = False)
#df.plot(kind = 'scatter')
plot_multi(data, figsize=(10, 5))
plt.xticks(np.arange(10), np.arange(1, 11))
plt.subplots_adjust(right=0.8)
plt.title(filename + ' Statistics')
plt.savefig(filename + '.png')
plt.show()    

