import random as rd
import sys

def split(rf, ofs_te, ofs_tr, num_split):
    for line in rf:
        idx = rd.randint(0,num_split-1)
        ofs_te[idx].write(line)
        for i in range(num_split):
            if i == idx:
                continue
            ofs_tr[i].write(line)

if __name__ == '__main__':
    num_split = 5
    rf = open(sys.argv[1], 'r')
    ofs_te = [ open("{}.{}.te".format(sys.argv[1],str(i)), 'w') for i in range(num_split)]
    ofs_tr = [ open("{}.{}.tr".format(sys.argv[1],str(i)), 'w') for i in range(num_split)]

    split(rf, ofs_te, ofs_tr, num_split)
