import numpy as np
import os, sys

item_num=1055
s=[0,0,0,0,0]
bids=np.empty((5,item_num))
for i,j in enumerate([0,3,4,5,6]):
    rng = np.random.RandomState(j)
    bids[i,:] = rng.gamma(20, 1/0.4, item_num)

np.random.seed(0)
with open(sys.argv[1], 'r') as gts:
    for gline in gts:
        gline = gline.strip()
        gt = gline.split(' ', 1)[0]
        gt = gt.split(':')[0]
        for i in range(5):
            if np.random.rand() < 0.9:
                s[i] += float(bids[i, int(gt)])
                #s += float(preds[1][i])*(0.9**(i+1))

print(s)
