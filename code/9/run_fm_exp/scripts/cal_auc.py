import numpy as np
import os, sys
import torch  
import tqdm
#from utility import recommend
from sklearn.metrics import roc_auc_score, log_loss
np.random.seed(0)

root = sys.argv[1]
gt_path = sys.argv[2]
pos_bias = float(sys.argv[3])

device = 'cpu'
batch_size_of_user = 500
rp = 10
#num_of_pos = 10
#res = np.empty(batch_size_of_user*num_of_pos, dtype=np.int32)

Qs = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Qva')])
Ps = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Pva')])
#print(Qs, Ps)
Qs = torch.tensor(np.vstack([np.expand_dims(np.load(i).T, axis=0) for i in Qs])).to(device)  # (n_fields,embed_dim,item_num) 
Ps = torch.tensor(np.hstack([np.expand_dims(np.load(i), axis=0) for i in Ps])).to(device)  # (n_fields,context_num,embed_dim)
#print(item_num, embed_dim)
#print(Qs.size(), Ps.size())

label_idxes, flags = list(), list()
with open(gt_path, 'r') as gt:
    pbar = tqdm.tqdm(gt, smoothing=0, mininterval=1.0)
    pbar.set_description('Loading gt:')
    for line in pbar:
        label, _ = line.strip().split(' ', 1)
        label = [tuple([int(i) for i in l.split(':')]) for l in label.split(',')]
        label_idx, flag = zip(*label)
        label_idxes.append(list(label_idx))  # (context_num, k)
        flags.append(list(flag)) # (context_num, k)
label_idxes = np.array(label_idxes)
flags = np.array(flags)
num_of_pos = flags.shape[1]
pos_bias = [pos_bias**i for i in range(num_of_pos)]
const_pos_bias = float(sum(pos_bias))/num_of_pos

def infer(label_idxes, Ps, Qs, batch_size_of_user, num_of_pos):
    predicts = list()
    item_num = Qs.size()[-1]
    embed_dim = Qs.size()[-2]
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, Ps.size()[1], batch_size_of_user), smoothing=0, mininterval=1.0):
            num_of_user = Ps[:, i:i+batch_size_of_user, :].size()[1]
            y = torch.sigmoid(torch.sum(torch.matmul(\
                    Ps[:, i:i+num_of_user, :].view(-1, 1, embed_dim), \
                    Qs[:, :, label_idxes.flatten()[i*num_of_pos:(i+num_of_user)*num_of_pos]]\
                    .view(-1, embed_dim, num_of_user, num_of_pos).transpose(1, 2)\
                    .view(-1, embed_dim, num_of_pos))\
                    .view(-1, num_of_user, num_of_pos), 0))  # (batch_size, num_of_pos)
            predicts.extend(torch.flatten(y).tolist())
    
    return predicts

def trans_flags(flags):
    const_flags = np.zeros_like(flags)
    pos_flags = np.zeros_like(flags)
    for i in range(flags.shape[0]):
        for j in range(flags.shape[1]):
            if flags[i, j] > 0:
                rnd = np.random.rand()
                if rnd < const_pos_bias:
                    const_flags[i, j] = flags[i, j]
                if rnd < pos_bias[j]:
                    pos_flags[i, j] = flags[i, j]

    return const_flags, pos_flags

def shuf(label_idxes, flags):
    p = np.random.permutation(flags.shape[1])
    for r in range(label_idxes.shape[0]):
        label_idxes[r, :] = label_idxes[r, p]
        flags[r, :] = flags[r, p]
    return

# real, pos, const
preds = infer(label_idxes, Ps, Qs, batch_size_of_user, num_of_pos)
const_flags, pos_flags = trans_flags(flags)
print('criteria real const pos')
print('auc',roc_auc_score(flags.flatten(), preds), roc_auc_score(const_flags.flatten(), preds), roc_auc_score(pos_flags.flatten(), preds))
print('logloss',log_loss(flags.flatten(), preds), log_loss(const_flags.flatten(), preds), log_loss(pos_flags.flatten(), preds))

duplc_preds = list()
duplc_flags = list()
duplc_const_flags = list()
duplc_pos_flags = list()
for i in range(rp):
    shuf(label_idxes, flags)
    preds = infer(label_idxes, Ps, Qs, batch_size_of_user, num_of_pos)
    const_flags, pos_flags = trans_flags(flags)
    duplc_preds.extend(preds)
    duplc_flags.extend(list(flags.flatten()))
    duplc_const_flags.extend(list(const_flags.flatten()))
    duplc_pos_flags.extend(list(pos_flags.flatten()))

print('criteria %d*real %d*const %d*pos'%(rp, rp, rp))
print('auc', roc_auc_score(duplc_flags, duplc_preds), roc_auc_score(duplc_const_flags, duplc_preds), roc_auc_score(duplc_pos_flags, duplc_preds))
print('logloss', log_loss(duplc_flags, duplc_preds), log_loss(duplc_const_flags, duplc_preds), log_loss(duplc_pos_flags, duplc_preds))

