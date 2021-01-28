import os, sys
import numpy as np
import shutil

SRC = sys.argv[1]
PORTION = float(sys.argv[2])
mode = sys.argv[3]

def process_helper(fin_path):
    rng = np.random.RandomState(0)
    with open(fin_path, 'r') as fin1, \
         open(fin_path.replace('random', 'det'), 'r') as fin2, \
         open(os.path.basename(fin_path).replace('random', 'select'), 'w') as trva:
        for i, line1 in enumerate(fin1):
            line1 = line1.strip()
            line2 = fin2.readline().strip()
            
            if rng.rand() < PORTION:
                tmp = line1  # rnd
                trva.write(tmp+'\n')
            elif mode == '.comb.':
                tmp = line2  # det
                trva.write(tmp+'\n')
            else:
                pass
    return

origins = [os.path.join(SRC, 'random_trva.ffm'), os.path.join(SRC, 'random_trva.svm')]
for o in origins:
    process_helper(o)

