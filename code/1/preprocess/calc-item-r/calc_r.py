import math
import sys

def logit(x):
    return math.log(x / (1.0-x))

if __name__ == '__main__':
    rf = open(sys.argv[1], 'r')
    app_cnt = {}
    pos_cnt = 0.0
    neg_cnt = 0.0
    for line in rf:
        toks = line.strip().split()
        labels = toks[0].strip(',').split(',')
        for label in labels:
            idx, val = map(int, label.split(':'))
            app_cnt.setdefault(idx, [0.0, 0.0])
            app_cnt[idx][val] += 1
            if val > 0:
                pos_cnt += 1
            else:
                neg_cnt += 1

    of = open(sys.argv[1]+".r_score", 'w')
    avg_ctr = pos_cnt / (neg_cnt + pos_cnt)
    max_key = 0.0
    for key in range(1000):
        item = app_cnt.get(key, [1,0])
        ctr = item[1] / ( item[0] + item[1] )
        if ctr > 0.0 and ctr < 1:
            of.write("{}\n".format(logit(ctr)))
        else:
            of.write("{}\n".format(logit(avg_ctr)))
        max_key = max(key, max_key)
    print(max_key, logit(avg_ctr))

