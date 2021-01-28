import torch
import torch.nn.functional as F

class FFM(torch.nn.Module):
    def __init__(self, inputSize, embed_dim):
        super().__init__()
        self.embed1 = torch.nn.Embedding(inputSize, embed_dim, padding_idx=0)  

        ## Pos
        #self.embed2 = torch.nn.Embedding(posSize+1, 1, padding_idx=0)  # set position 0th as padding idx, real position starts from 1 to 10

        torch.nn.init.xavier_uniform_(self.embed1.weight.data[1:, :])
        #torch.nn.init.xavier_uniform_(self.embed2.weight.data[1:, :])


    def forward(self, x1, x2, x3, x4):  # x1: context, x2: item, x3: position, x4: context value
        x1 = torch.sum(torch.mul(self.embed1(x1), x4.unsqueeze(2)), dim=1)  # field 1 embedding for cxt: (batch_size, cxt_nonzero_feature_num, embed_dim)
        x2 = torch.sum(self.embed1(x2), dim=1)  # field 1 embedding for item: (batch_size, item_nonzero_feature_num, embed_dim)

        ## merge
        x12 = torch.sigmoid(torch.sum(x1*x2, dim=1))  # (batch_size,)
        #x3 = torch.sum(self.embed2(x3), dim = 1)  # (batch_size,)
        #x3 = torch.sigmoid(x3).squeeze(1) 
        #out = x12*x3  # ffm_prob*pos_prob
        return x12
