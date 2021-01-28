import os, sys

root = sys.argv[1]
flag = 1 if sys.argv[2] == 'auc' else 0
log_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('log')]

records = {}
for lp in log_paths:
    records[lp] = [0., 1000., 0.] # iter, min_logloss, max_auc
    with open(lp) as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            logloss = float(line[-1].split(':')[-1])
            auc = float(line[-2].split(':')[-1])
            iter_num = int(line[0].split(':')[-1])

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

#print(params)
tmp = [float(i.split('-', 1)[-1]) for i in os.path.basename(params[0]).split('_')[1:5]]
print(' '.join([str(i) for i in tmp + params[1]]))


