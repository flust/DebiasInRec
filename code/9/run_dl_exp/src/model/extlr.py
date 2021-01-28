import torch
import torch.nn.functional as F

class ExtLogisticRegression(torch.nn.Module):
    def __init__(self, inputSize1, inputSize2):
        super().__init__()
        #self.fc1 = torch.nn.Linear(inputSize1, 1, bias=True)
        #self.fc2 = torch.nn.Linear(inputSize2, 1, bias=True)
        self.fc1 = torch.nn.Embedding(inputSize1, 1, padding_idx=0)
        self.bias1 = torch.nn.Parameter(torch.zeros((1,)))
        self.fc2 = torch.nn.Embedding(inputSize2+1, 1, padding_idx=0) # add 1 for padding_idx
        #self.bias2 = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, x1, x2):
        x1 = torch.sum(self.fc1(x1), dim = 1) + self.bias1
        x2 = torch.sum(self.fc2(x2), dim = 1)
        out = torch.sigmoid(x1+x2)
        return out.squeeze(1)
