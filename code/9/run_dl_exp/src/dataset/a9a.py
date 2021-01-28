import os
import time
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_svmlight_file

class A9ADataset(Dataset):
    def __init__(self, root_dir, training=True, transform=None):
        self.items = []
        self.contexts = []
        self.labels = []
        self.len = 0  # number of samples
        self.n_feature = 0  # dim of features

        data_file = 'a9a'
        data_file = os.path.join(root_dir, data_file)
        with open(data_file, 'r') as data:
            for line in data.readlines():
                line = line.strip()
                l, c = line.split(' ', 1)
                l = 0 if l == '-1' else 1 
                c = sorted([int(i.split(':')[0]) for i in c.split(' ')])
                self.labels.append(l)
                self.contexts.append(c)
                n_feature = c[-1]
                if n_feature > self.n_feature:
                    self.n_feature = n_feature
        self.n_feature += 1    # drop feature not in trainset  
        print('max dim:%d'%self.n_feature)
    
    def __len__(self):
        return len(self.contexts) # 10 ads per context

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        
        context = self.contexts[idx]
        label = self.labels[idx]
        data = context

        return {'data':data, 'label':label, 'pos':1}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = PositionDataset('../../data/random', training=False)
    #data = dataset.__getitem__(102*10+7)
    #print(sum(data['data']), data['label'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    print('Start loading!')
    for i_batch, sample_batched in enumerate(dataloader):
        if 1 in sample_batched['label']:
            print(i_batch, sample_batched)
            break
