import numpy as np
import torch
import torch.nn.functional as F

class FeaturesLinear(torch.nn.Module):

    # field_dims: a list of each lenth of each field
    def __init__(self, input_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(input_dims, output_dim, padding_idx=0)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x1, x2):
        """
        :param x1: Long tensor of size ``(batch_size, num_itemFeatures)``
        :param x2: Long tensor of size ``(batch_size, num_contextFeatures)``
        :output: Long tensor of size ``(batch_size, output_dims)``
        """
        #x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x1), dim=1) + torch.sum(self.fc(x2), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, input_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dims, embed_dim, padding_idx=0)
        #self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data[1:, :])

    def forward(self, x1, x2):
        """
        :param x1: Long tensor of size ``(batch_size, num_itemFeatures)``
        :param x2: Long tensor of size ``(batch_size, num_contextFeatures)``
        :output: Long tensor of size ``(batch_size, 2, embed_dims)``
        """
        #x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x1 = self.embedding(x1).mean(dim=1, keepdim=True)
        x2 = self.embedding(x2).mean(dim=1, keepdim=True)

        return torch.cat((x1, x2), 1)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields*embed_dim)``
        """
        return self.mlp(x)


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for cross_layer_size in cross_layer_sizes:
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        #print(x0.size(), h.size())
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half: #and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2  # [batch_size, embed_dim]
        sum_of_square = torch.sum(x ** 2, dim=1)  # []
        ix = square_of_sum - sum_of_square  # (batch_size, embed_dim)
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)  # (batch_size)
        return 0.5 * ix


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]  # (num_fields, feature_num, embed_dim)
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])  # [filed jth, :, embed_dim ith] 
        ix = torch.stack(ix, dim=1)
        return ix


