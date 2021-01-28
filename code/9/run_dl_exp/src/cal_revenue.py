import numpy as np
import os, sys
import pickle
from collections import defaultdict as ddict
from sklearn.metrics import roc_auc_score, log_loss
np.random.seed(0)
try:
    bias_base = float(sys.argv[3])
except:
    bias_base = 0.9

pos_biases = [bias_base**i for i in range(10)]
const_pos_bias = sum(pos_biases)/10.
probs = list()
probs2 = list()
truths = list()
pos_truths = list()
const_truths = list()
stats = ddict(int)

with open(sys.argv[1], 'r') as preds, open(sys.argv[2], 'r') as gts:
    revenue = 0.
    count = 0.
    for pline in preds:
        count+=1.
        gline = gts.readline().strip()
        pline = pline.strip()
        gt = gline.split(' ', 1)[0]
        gt = gt.split(':')[0]
        preds = [tuple(p.split(':')) for p in pline.split(' ')]
        preds = list(zip(*preds))  # (ad, prob, bid) sorted by bid*prob
        probs.extend([float(p) for p in preds[1]])

        idxes = np.arange(10)
        np.random.shuffle(idxes)

        for i, k in enumerate(zip(preds[0], idxes)):
            ad, idx = k
            #stats[ad] += float(preds[1][i])
            rnd = np.random.rand()

            if gt == ad:
                truths.append(1)
                if rnd <= pos_biases[i]:
                    stats[ad] += float(preds[-1][i])
                    revenue += float(preds[-1][i])
            else:
                truths.append(0)

            ad2 = preds[0][idx]
            probs2.append(float(preds[1][idx]))
            if gt == ad2:
                if rnd <= pos_biases[i]:
                    pos_truths.append(1)
                else:
                    pos_truths.append(0)
                if rnd <= const_pos_bias:
                    const_truths.append(1)
                else:
                    const_truths.append(0)
            else:
                pos_truths.append(0)
                const_truths.append(0)

print(revenue)
#for i in sorted(stats.items(), key=lambda x: int(x[0])):
#    #print('%s,%.6f'%(i[0], i[1]/count))
#    print('%s,%.6f'%(i[0], i[1]))
#print(roc_auc_score(truths, probs), log_loss(truths, probs))
#print(roc_auc_score(const_truths, probs2), log_loss(const_truths, probs2))
#print(roc_auc_score(pos_truths, probs2), log_loss(pos_truths, probs2))
#print(sorted(stats.items(), key=lambda x:x[-1], reverse=True))
#pickle.dump(stats, open('revenue.stats', 'wb'))
