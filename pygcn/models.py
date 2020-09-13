import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
from torch.nn.init import xavier_normal_, xavier_uniform_
import math
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output_dim, dropout, adj, n_nodes, args=None):
        super(GCN, self).__init__()
        self.embeddings = nn.Embedding(n_nodes, nfeat)
        self.loss = nn.BCELoss()
        self.gc1 = GraphConvolution(nfeat, nhid, adj=adj)
        self.gc2 = GraphConvolution(nhid, output_dim, adj=adj)
        self.dropout = dropout
        self.inp_drop = nn.Dropout(args.input_dropout)
        self.feature_map_drop = nn.Dropout(dropout)
        self.hidden_drop = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(2, args.channels, args.kernel_size, stride=1, padding=int(math.floor(args.kernel_size/2)))
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(args.channels)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.fc = nn.Linear(output_dim * args.channels, output_dim)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.bn4 = nn.BatchNorm1d(output_dim)
        self.bn_init = nn.BatchNorm1d(nfeat)
        self.batch_size = args.batch_size
        # self.training = args.training

    
    def init(self):
        xavier_normal_(self.embeddings.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)


    def forward(self, e1, rel, X):
        e1 = e1.view(-1, 1)
        rel = rel.view(-1, 1)
        emb_initial = self.embeddings(X)
        x = self.gc1(emb_initial)
        x = self.bn3(x)
        x = F.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.bn4(self.gc2(x))
        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, self.dropout, training=self.training)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = e1_embedded_all[rel]
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(self.batch_size, -1)
        import pdb
        pdb.set_trace()
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)
        return pred

        # emb1 = self.embeddings(index)
        # x = F.relu(self.gc1(emb1))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x)
        # return x
