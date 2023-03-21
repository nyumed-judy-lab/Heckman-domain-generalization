'''
TabNet

Attention Network


'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularSelfAttention(nn.Module):
    def __init__(self, n_features, n_heads=4, hidden_size=128):
        super(TabularSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = hidden_size * n_heads
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.d_model)
        )
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        # x shape: [batch_size, n_features]
        x = self.projection(x)
        batch_size = x.size(0)
        # reshape x to [batch_size, n_heads, n_features_per_head]
        x = x.view(batch_size, self.n_heads, self.hidden_size)
        # compute queries, keys, and values
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # compute attention weights
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        weights = F.softmax(scores, dim=-1)
        # apply attention weights to values
        outputs = torch.matmul(weights, v).view(batch_size, -1)
        return outputs



class DecisionTree(nn.Module):
    def __init__(self, input_size, output_size):
        super(DecisionTree, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.fc3 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = torch.sigmoid(self.fc1(x))
        out1 = self.fc2(x)
        out2 = torch.sigmoid(self.fc3(x))
        out = out1 * out + (1 - out1) * out2
        return out

class DNDF(nn.Module):
    def __init__(self, input_size, output_size, num_trees=10):
        super(DNDF, self).__init__()
        self.trees = nn.ModuleList([DecisionTree(input_size, output_size) for i in range(num_trees)])
        
    def forward(self, x):
        out = torch.cat([tree(x) for tree in self.trees], dim=1)
        out = F.softmax(out, dim=1)
        return out