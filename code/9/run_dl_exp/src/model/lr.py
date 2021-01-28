import torch
import torch.nn.functional as F

def simple_elementwise_apply(fn, packed_sequence):
    """applies a pointwise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

class LogisticRegression(torch.nn.Module):
    def __init__(self, inputSize):
        super().__init__()
        self.linear = torch.nn.Embedding(inputSize, 1, padding_idx=0)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        #out = torch.sum(simple_elementwise_apply(self.linear, x).data, dim = 1) + self.bias
        out = torch.sum(self.linear(x), dim = 1) + self.bias
        return torch.sigmoid(out.squeeze(1))
