import os, sys

fin_path = sys.argv[1]
#fout_path = fin_path.replace('random', 'trva')
fout_path = fin_path.replace(sys.argv[2], sys.argv[2]+'_trva')
with open(fin_path, 'r') as fin, open(fout_path, 'w') as fout:
    for line in fin:
        line = line.strip()
        labels, features = line.split(' ', 1)
        new_labels = list()
        for l in labels.split(','):
            l = l.strip().split(':')[:2]
            l = ':'.join(l)
            new_labels.append(l)
        new_labels = ','.join(new_labels)
        fout.write('%s %s\n'%(new_labels, features))

        
