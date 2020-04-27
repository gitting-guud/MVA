# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:42:06 2020

@author: Houcine's laptop
"""

import argparse
from os import path

import numpy as np
import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader


MODEL_STATE_FILE = path.join("model_state_1.pth")


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    
    
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs), dim=0)
        
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, n_layers):
        super(GAT, self).__init__()
#        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
#        # Be aware that the input dimension is hidden_dim*num_heads since
#        # multiple head outputs are concatenated together. Also, only
#        # one attention head in the output layer.
#        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
#        self.layers = [self.layer1, self.layer2]
        
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads))
        for i in range(n_layers-2):
            self.layers.append(MultiHeadGATLayer(g, hidden_dim * num_heads, hidden_dim, num_heads))
        self.layers.append(MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 6, merge="mean"))
        # self.layers.append(MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1))
        

    def forward(self, h):
        outputs = h
        for i, layer in enumerate(self.layers):
            if i == len(self.layers)-1 :
                outputs = layer(outputs)
            else :
                outputs = F.elu(layer(outputs))
        return outputs


def main(args):
    # create the dataset
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))
#    model = BasicGraphModel(g=train_dataset.graph, n_layers=2, input_size=n_features,
#                            hidden_size=256, output_size=n_classes, nonlinearity=F.elu).to(device)

    model = GAT(  g=train_dataset.graph,
                  in_dim=n_features,
                  hidden_dim=256,
                  out_dim=n_classes,
                  num_heads=4,
                  n_layers=2
               ).to(device)
    
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-2)

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE))
    return test(model, loss_fcn, device, test_dataloader)


def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.layers:
                for head in layer.heads :
                    head.g = subgraph
                layer.g = subgraph
            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 5 == 0:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                features = torch.tensor(features).to(device)
                labels = torch.tensor(labels).to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))


def test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores


def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
            for head in layer.heads :
                head.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


def collate_fn(sample):
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    main(args)
