import os, sys

root = sys.argv[1]
flag = 1 if sys.argv[2] == 'auc' else 0
log_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('log')]

records = {}
for lp in log_paths:
    records[lp] = [0., 1000., 0.] # iter, min_logloss, max_auc
    with open(lp) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            line = line.strip().split(' ')
            line = [s for s in line if s != '']
            iter_num = float(line[0])
            logloss = float(line[-2])
            auc = float(line[-1])

            if flag:
                if auc > records[lp][-1]:
                    records[lp][0] = iter_num
                    records[lp][1] = logloss
                    records[lp][2] = auc
            else:
                if logloss < records[lp][1]:
                    records[lp][0] = iter_num
                    records[lp][1] = logloss
                    records[lp][2] = auc

if flag:
    params = sorted(records.items(), key=lambda x: x[-1][-1], reverse=flag)[0]
else:
    params = sorted(records.items(), key=lambda x: x[-1][-2], reverse=flag)[0]

print(params[0].split('/')[-1].split('.')[0], params[0].split('/')[-1].split('.')[2], int(params[1][0]), params[1][1], params[1][2],)

