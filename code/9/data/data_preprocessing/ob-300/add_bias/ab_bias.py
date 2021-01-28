import sys
import random as rd

POS = 10
base_rate = 0.5

file=sys.argv[1]

def init_pos_bias( gamma ):
    res_list = []
    alpha_k = 1
    for i in range(POS):
        res_list.append( gamma * alpha_k )
        alpha_k *= base_rate
    return res_list

def change_label_to_zero( tk ):
    idx, label, prop = tk.split(":")
    return "{}:{}:{}".format(idx, '0', prop)

def output_bias_file( file_name ):
    ofile_name = "{}.pos.{}.bias".format(file_name, base_rate)
    rf = open(file_name, 'r')
    of = open(ofile_name, 'w')
    pos_bias_list = init_pos_bias( 1.0 )
    print(pos_bias_list)
    for line in rf:
        labels = line.strip().split()[0]
        feats = line.strip().split()[1:]
        toks = labels.strip().strip(',').split(',')
        len(toks)
        for i, tk in enumerate(toks):
            if rd.random() >= pos_bias_list[i]:
                toks[i] = change_label_to_zero(tk)
            else:
                idx, label, prop = tk.split(":")
                toks[i] = "{}:{}:{}".format(idx, label, prop)
        of.write("{} {}\n".format(','.join(toks), ' '.join(feats)))

if __name__ == '__main__':
    output_bias_file(file)
