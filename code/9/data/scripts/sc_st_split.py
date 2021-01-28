import os, sys
import numpy as np
import shutil

SRC = sys.argv[1]
PORTION = float(sys.argv[2])

def drop_prop(line):
    line = line.split(' ', 1)
    line[0] = ','.join([':'.join(label.split(':')[:-1]) for label in line[0].strip().split(',')])
    tmp = ' '.join(line)
    return tmp

def process_helper(fin_path):
    rng = np.random.RandomState(0)
    with open(fin_path, 'r') as fin1, \
         open(fin_path.replace('random', 'det'), 'r') as fin2, \
         open(os.path.basename(fin_path).replace('random', 'select_sc'), 'w') as tr_sc, \
         open(os.path.basename(fin_path).replace('random', 'select_st'), 'w') as tr_st:
        for i, line1 in enumerate(fin1):
            line1 = line1.strip()
            line2 = fin2.readline().strip()
            if rng.rand() < PORTION:
                tmp = line1  # rnd
                #tmp = drop_prop(tmp)
                tr_st.write(tmp+'\n')
            else:
                tmp = line2  # det
                #tmp = drop_prop(tmp)
                tr_sc.write(tmp+'\n')

    return

origins = [os.path.join(SRC, 'random_trva.ffm'), os.path.join(SRC, 'random_tr.ffm')]
for o in origins:
    process_helper(o)

