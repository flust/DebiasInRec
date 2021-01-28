import numpy as np
import os, sys
import torch  
import tqdm
from utility import recommend

root = sys.argv[1]
device = 'cpu'
batch_size_of_user = 50
num_of_pos=10
res = np.empty(batch_size_of_user*num_of_pos, dtype=np.int32)

Qs = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Qva')])
Ps = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Pva')])
#print(Qs, Ps)
Qs = torch.tensor(np.vstack([np.expand_dims(np.load(i).T, axis=0) for i in Qs])).to(device)
Ps = torch.tensor(np.vstack([np.expand_dims(np.load(i), axis=0) for i in Ps])).to(device)
item_num = Qs.size()[-1]
total_user_num=Ps.size()[1]
#print(Qs.size(), Ps.size())

rngs = [np.random.RandomState(seed) for seed in [0,3,4,5,6]]
bids = np.empty((len(rngs), item_num)) 
for i, rng in enumerate(rngs):
    #bids[i, :] = rng.gamma(1.5, 10, item_num)
    bids[i, :] = rng.gamma(10, 0.4, item_num)
    #bids[i, :] = np.array([2**i for i in range(item_num)])
    #bids[i, :] = np.ones(item_num)
bids = torch.tensor(bids).to(device)

ctr_sum = np.zeros(item_num)
with torch.no_grad():
    fs = list()
    for j in range(len(rngs)):
        fs.append(open(os.path.join(root, 'tmp.pred.%d'%j), 'w'))
    for i in tqdm.tqdm(range(0, Ps.size()[1], batch_size_of_user), smoothing=0, mininterval=1.0):
        y = torch.sigmoid(torch.sum(torch.matmul(Ps[:, i:i+batch_size_of_user, :], Qs), 0))
        #origin_y = torch.sigmoid(torch.sum(torch.matmul(Ps[:, i:i+batch_size_of_user, :], Qs), 0))
        #for n in range(y.size()[0]):
        #    print(",".join(["%f"%m for m in y[n, :].numpy()]))
        #ctr_sum += torch.sum(y, 0).numpy()
    #for i in ctr_sum:
        #print(i/total_user_num)

        num_of_user = y.size()[0]
        for j in range(len(rngs)):
            fp = fs[j]
            out = y.flatten()*(bids[j, :].repeat(num_of_user))
            recommend.get_top_k_by_greedy(out.cpu().numpy().flatten(), num_of_user, item_num, num_of_pos, res[:num_of_user*num_of_pos])
            _res = res[:num_of_user*num_of_pos].reshape(num_of_user, num_of_pos)
            for r in range(num_of_user):
                tmp = ['%d:%.4f:%0.4f'%(ad, y[r, ad], bids[j, ad]) for ad in _res[r, :]]
                #tmp = ['%d:%.4f'%(ad, bids[j, ad]) for ad in _res[r, :]]
                #tmp = ["%d:%.4f"%(ad, origin_y.cpu().numpy()[r, ad]) for ad in _res[r, :]]
                #tmp = ["%d:%.4f"%(ad, r) for ad in _res[r, :]]
                #tmp = ['%d:%f'%(ad, m) for m in y[r, :]]
                fp.write('%s\n'%(' '.join(tmp)))
            #break
        #break
    for j in range(len(rngs)):
        fs[j].close()
 
