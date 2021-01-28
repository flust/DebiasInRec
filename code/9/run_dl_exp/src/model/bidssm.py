import torch
import torch.nn.functional as F

class BiDSSM(torch.nn.Module):
    def __init__(self, inputSize, embed_dim, posSize):
        super().__init__()
        self.embed = torch.nn.Embedding(inputSize, embed_dim, padding_idx=0)  # trans one-hot vector to 300 dimensions
        ## Tower 1 for context feature
        self.t1_bias1 = torch.nn.Parameter(torch.zeros((embed_dim,)))
        self.t1_fc2 = torch.nn.Linear(embed_dim, embed_dim, bias=True)  # add 1 for padding_idx
        #self.t1_fc3 = torch.nn.Linear(100, embed_dim, bias=True)  # add 1 for padding_idx

        ## Tower 2 for item feature
        self.t2_bias1 = torch.nn.Parameter(torch.zeros((embed_dim,)))
        self.t2_fc2 = torch.nn.Linear(embed_dim, embed_dim, bias=True)  # add 1 for padding_idx
        #self.t2_fc3 = torch.nn.Linear(100, embed_dim, bias=True)  # add 1 for padding_idx

        ## Pos
        self.embed2 = torch.nn.Embedding(posSize+1, 1, padding_idx=0)  # trans one-hot vector to 300 dimensions

        ## Init
        torch.nn.init.xavier_uniform_(self.embed1.weight.data[1:, :])
        torch.nn.init.xavier_uniform_(self.embed2.weight.data[1:, :])
        self.embed2.weight.data[0, :] = float(10000)


    def forward(self, x1, x2, x3, x4, use_relu=False):
        if use_relu:
            act = torch.relu
        else:
            act = torch.tanh
        ## Tower 1
        #x1 = torch.sum(self.embed(x1), dim = 1) + self.t1_bias1
        x1 = torch.sum(torch.mul(self.embed(x1), x4.unsqueeze(2)), dim = 1) + self.t1_bias1
        x1 = act(x1)
        x1 = self.t1_fc2(x1) 
        x1 = act(x1)
        #x1 = self.t1_fc3(x1) 
        #x1 = act(x1)
        ## Tower 2 
        x2 = torch.sum(self.embed(x2), dim = 1) + self.t2_bias1
        x2 = act(x2)
        x2 = self.t2_fc2(x2) 
        x2 = act(x2)
        #x2 = self.t2_fc3(x2) 
        #x2 = act(x2)
        ## merge
        x12 = torch.sigmoid(torch.sum(x1*x2, dim = 1))
        x3 = torch.sum(self.embed2(x3), dim = 1)
        x3 = torch.sigmoid(x3).squeeze(1)
        out = x12*x3
        return out
