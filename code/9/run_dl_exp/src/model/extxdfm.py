import torch
import torch.nn.functional as F
import numpy as np

from src.model.layer import CompressedInteractionNetwork, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


#class FeaturesLinear(torch.nn.Module):
#
#    # field_dims: a list of each lenth of each field
#    def __init__(self, input_dims, output_dim=1):
#        super().__init__()
#        self.fc = torch.nn.Embedding(input_dims, output_dim, padding_idx=0)
#        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
#
#    def forward(self, x1, x2):
#        """
#        :param x1: Long tensor of size ``(batch_size, num_itemFeatures)``
#        :param x2: Long tensor of size ``(batch_size, num_contextFeatures)``
#        :output: Long tensor of size ``(batch_size, output_dims)``
#        """
#        #x = x + x.new_tensor(self.offsets).unsqueeze(0)
#        return torch.sum(self.fc(x1), dim=1) + torch.sum(self.fc(x2), dim=1) + self.bias
#
#
#class FeaturesEmbedding(torch.nn.Module):
#
#    def __init__(self, input_dims, embed_dim):
#        super().__init__()
#        self.embedding = torch.nn.Embedding(input_dims, embed_dim, padding_idx=0)
#        #self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
#        #torch.nn.init.xavier_uniform_(self.embedding.weight.data)
#
#    def forward(self, x1, x2):
#        """
#        :param x1: Long tensor of size ``(batch_size, num_itemFeatures)``
#        :param x2: Long tensor of size ``(batch_size, num_contextFeatures)``
#        :output: Long tensor of size ``(batch_size, 2, embed_dims)``
#        """
#        #x = x + x.new_tensor(self.offsets).unsqueeze(0)
#        x1 = self.embedding(x1).mean(dim=1, keepdim=True)
#        x2 = self.embedding(x2).mean(dim=1, keepdim=True)
#
#        return torch.cat((x1, x2), 1)
#
#
#class MultiLayerPerceptron(torch.nn.Module):
#
#    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
#        super().__init__()
#        layers = list()
#        for embed_dim in embed_dims:
#            layers.append(torch.nn.Linear(input_dim, embed_dim))
#            layers.append(torch.nn.BatchNorm1d(embed_dim))
#            layers.append(torch.nn.ReLU())
#            layers.append(torch.nn.Dropout(p=dropout))
#            input_dim = embed_dim
#        if output_layer:
#            layers.append(torch.nn.Linear(input_dim, 1))
#        self.mlp = torch.nn.Sequential(*layers)
#
#    def forward(self, x):
#        """
#        :param x: Float tensor of size ``(batch_size, num_fields*embed_dim)``
#        """
#        return self.mlp(x)
#
#
#class CompressedInteractionNetwork(torch.nn.Module):
#
#    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
#        super().__init__()
#        self.num_layers = len(cross_layer_sizes)
#        self.split_half = split_half
#        self.conv_layers = torch.nn.ModuleList()
#        prev_dim, fc_input_dim = input_dim, 0
#        for i, cross_layer_size in enumerate(cross_layer_sizes):
#            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
#                                                    stride=1, dilation=1, bias=True))
#            if self.split_half and i != self.num_layers - 1:
#                cross_layer_size //= 2
#            prev_dim = cross_layer_size
#            fc_input_dim += prev_dim
#        self.fc = torch.nn.Linear(fc_input_dim, 1)
#
#    def forward(self, x):
#        """
#        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
#        """
#        xs = list()
#        x0, h = x.unsqueeze(2), x
#        #print(x0.size(), h.size())
#        for i in range(self.num_layers):
#            x = x0 * h.unsqueeze(1)
#            batch_size, f0_dim, fin_dim, embed_dim = x.shape
#            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
#            x = F.relu(self.conv_layers[i](x))
#            if self.split_half and i != self.num_layers - 1:
#                x, h = torch.split(x, x.shape[1] // 2, dim=1)
#            else:
#                h = x
#            xs.append(x)
#        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class ExtExtremeDeepFactorizationMachineModel(torch.nn.Module):
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

    def forward(self, x1, x2, x3):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x1, x2)
        #print(embed_x.size())
        x = self.linear(x1, x2) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        x3 = torch.sum(self.embed2(x3), dim = 1)  # (batch_size,)
        return torch.sigmoid(x.squeeze(1) + x3.squeeze(1))
