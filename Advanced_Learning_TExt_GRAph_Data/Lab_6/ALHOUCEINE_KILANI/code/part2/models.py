"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        # your code here #
        ##################
        output = self.fc1(x_in)
        h1 = torch.mm(adj, output)
        Z0 = self.relu(h1)
        
        Z0 = self.dropout(Z0)
        
        output = self.fc2(Z0)
        h2 = torch.mm(adj, output)
        Z1_before_do = self.relu(h2)
        
        Z1 = self.dropout(Z1_before_do)
        
        x = self.fc3(Z1)
        
        return F.log_softmax(x, dim=1), Z1_before_do 