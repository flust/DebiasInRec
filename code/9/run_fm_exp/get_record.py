import os, sys

root = sys.argv[1]
mode = sys.argv[2]

f_path = os.path.join(root, "%s.record"%mode)
va_num = [2] 
te_num = list(range(4, 25, 4)) + list(range(5, 26, 4))
logloss = list()
auc = list()
with open(f_path, 'r') as f:
    for i, line in enumerate(f):
        if i+1 in va_num:
            logloss.append(line.strip().split(' ')[0])
            auc.append(line.strip().split(' ')[1])
        if i+1 in te_num:
            if (i+1) % 2 == 0:
                logloss.append(line.strip().split(' ')[-1])
            else:
                auc.append(line.strip().split(' ')[-1])

for i in logloss:
    print(i)
for i in auc:
    print(i)
