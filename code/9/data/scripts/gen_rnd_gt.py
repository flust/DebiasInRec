import os, sys
import numpy as np
np.random.seed(0)

fin_path = sys.argv[1]
fout_path = sys.argv[2]
item_num = int(sys.argv[3])
try:
    mode = sys.argv[4]
except:
    mode = '.' 
pos = float(sys.argv[5])

portion = sum([pos**i for i in range(10)])/10.
items = [str(i) for i in range(item_num)]
with open(fin_path, 'r') as fin, open(fout_path, 'w') as fout:
    for line in fin:
        line = line.strip()
        label, feature = line.split(' ', 1)
        label = label.split(':')[0]
        rnd = np.random.rand()
        choices = np.random.choice(items, 10, replace=False)
        tmp = list()
        for i,c in enumerate(choices):
            if c == label:
                if mode == '.':
                    tmp.append(c+':1')
                elif mode == '.const.':
                    if rnd <= portion:
                        tmp.append(c+':1')
                    else:
                        tmp.append(c+':0')
                elif mode == '.pos.':
                    if rnd <= pos**i:
                        tmp.append(c+':1')
                    else:
                        tmp.append(c+':0')
                else:
                    raise
            else:
                tmp.append(c+':0')
        tmp = ','.join(tmp)
        fout.write('%s %s\n'%(tmp, feature))


        

