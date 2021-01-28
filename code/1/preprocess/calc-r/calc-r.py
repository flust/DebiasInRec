import math
import sys

def logit(x):
    print(x)
    return math.log(x / (1.0-x))

if __name__ == '__main__':
    rf = open(sys.argv[1], 'r')
    pos_cnt = 0.0
    neg_cnt = 0.0
    for line in rf:
        toks = line.strip().split()
        labels = toks[0].strip(',').split(',')
        for label in labels:
            idx, val = map(int, label.split(':'))
            if val == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1
    print( neg_cnt, pos_cnt )
    ctr = pos_cnt / (neg_cnt + pos_cnt)
    print( ctr, logit(ctr))
