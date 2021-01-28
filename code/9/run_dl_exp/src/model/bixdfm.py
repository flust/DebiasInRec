import torch
import torch.nn.functional as F
import numpy as np

from src.model.layer import CompressedInteractionNetwork, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class BiExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch varietal implementation of xDeepFM, which only have 2 fields

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, input_dims, pos_dim, embed_dim=32, mlp_dims=(16, 16), dropout=0.2, cross_layer_sizes=(16, 16), split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(input_dims, embed_dim)
        self.embed2 = torch.nn.Embedding(pos_dim+1, 1, padding_idx=0)
        self.embed_output_dim = 2 * embed_dim  # only 2 fields: context and item
        self.cin = CompressedInteractionNetwork(2, cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(input_dims)

        torch.nn.init.xavier_uniform_(self.embed2.weight.data[1:, :])
        self.embed2.weight.data[0, :] = float(10000)

    def forward(self, x1, x2, x3):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x1, x2)
        #print(embed_x.size())
        x = self.linear(x1, x2) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        x3 = torch.sum(self.embed2(x3), dim = 1)
        x3 = torch.sigmoid(x3).squeeze(1)

        return torch.sigmoid(x.squeeze(1)) * x3
