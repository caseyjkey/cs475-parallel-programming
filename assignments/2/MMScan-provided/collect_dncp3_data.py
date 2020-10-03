#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

program_name = './MMScanDNCP3'
arguments = '50000 6'
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


mean_times = {} # [[threads, mean], ...]
aux_list = [1, 10, 100, 1000, 10000, 40000, 60000, 100000]
for aux in aux_list:
    print("Threads: 16")
    os.environ['OMP_NUM_THREADS'] = '16'
    mean_times[aux] = []
    for p in [40000, 60000, 100000, 300000, 500000]:
        results = []
        for i in range(0, 8): 
            output = subprocess.run([program_name] + (arguments + ' ' + str(aux)).split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
            result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
            print(result)
            results.append(float(result))
        results.remove(max(results))
        results.remove(min(results))
        mean = sum(results)/len(results)
        print('Aux:', aux, 'P:', p, 'Mean:', mean, '\n')
        mean_times[aux].append((p, mean))

best_times = []
for aux in aux_list:
    lowest = (69, 1000000000)
    for i, datum in enumerate(mean_times[aux]):
        p = datum[0]
        mean = datum[1]
        if mean < lowest[1]:
            lowest = datum

    best_times.append((aux, lowest[0], lowest[1]))
        

data = pd.DataFrame({
    "Aux": aux_list,
    "P": [datum[1] for datum in best_times],
    "Mean Execution Time (s)": [datum[2] for datum in best_times]
#    "Speedup (%)": [datum[1] for datum in speedup_list],
#    "Efficiency (%)": [datum[1] for datum in efficiency_list]
})

filename = program_name.split('/')[1] 
data.to_csv(filename + '.csv', index = False)
#data.plot(x='Aux', y='Mean Execution Time (s)', kind = 'scatter')
plot_multi(data, figsize=(10, 5))
#plt.xticks(np.arange(8), np.arange(1, 9))
#plt.subplots_adjust(right=0.8)
plt.title(filename + ' Statistics')
plt.savefig(filename + '.png')
plt.show()    

