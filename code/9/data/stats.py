import os, sys
import pickle
import numpy as np
import tqdm
from easydict import EasyDict as edict

stats = edict()
count = 0.
for i in range(int(sys.argv[3])):
    stats[str(i)] = np.zeros((2,10))
    
with open(sys.argv[1], 'r') as f:
    for line in tqdm.tqdm(f):
        line = line.strip()
        labels = [i for i in line.split(' ', 1)[0].split(',')]
        for pos, l in enumerate(labels):
            item, click = tuple(l.split(':')[:2])
            if item not in stats:
                raise
                #stats[item] = np.zeros((2,10))
            stats[item][0, pos] += int(click)
            stats[item][1, pos] += 1

with open(sys.argv[2], 'w') as f:
    #f.write('ad,%s,%s\n'%(','.join(['click_%d'%i for i in range(10)]+['count_%d'%i for i in range(10)]), "oCTR"))
    f.write('ad,%s\n'%(','.join(['click_%d'%i for i in range(10)]+['count_%d'%i for i in range(10)])))
    for i in sorted(stats.items(), key=lambda x: int(x[0])):
        f.write('%s,%s\n'%(i[0],','.join([str(j) for j in i[1].flatten()])))
#pickle.dump(stats, open('trva.stats', 'wb'))
