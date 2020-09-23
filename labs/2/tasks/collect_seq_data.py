#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

program_name = './Merge_sort_SEQ'
arguments = '1000'
time_token = 'time ='

mean_times = [] # [[threads, mean], ...]
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

data = pd.DataFrame({
    "Execution Time (s)": results,
    "Iteration": [num for num in range(0,6)]
})

filename = program_name.split('/')[1] 
data.to_csv(filename + '.csv', index = False)
#df.plot(kind = 'scatter')
data.plot(x='Iteration', y='Execution Time (s)', kind = 'scatter', figsize=(10, 5))
plt.title(filename + ' Statistics')
plt.savefig(filename + '.png')
plt.show()    
