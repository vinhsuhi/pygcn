from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, create_adj
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--train_path', type=str, default='data/FB15k-237/train.txt')
parser.add_argument('--val_path', type=str, default='data/FB15k-237/valid.txt')
parser.add_argument('--test_path', type=str, default='data/FB15k-237/test.txt')
parser.add_argument('--n_feats', type=int, default=200)
parser.add_argument('--input_dropout', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--channels', type=int, default=200)
parser.add_argument('--kernel_size', type=int, default=5)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# TODO: load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
entity_set, relation_set, train_triples, val_triples, test_triples = load_data(args.train_path, args.test_path, args.val_path)
adj, id2idx = create_adj(train_triples, entity_set, relation_set)
train_triple_ids = [[id2idx[ele] for ele in triple] for triple in train_triples]
test_triple_ids = [[id2idx[ele] for ele in triple] for triple in test_triples]
val_triple_ids = [[id2idx[ele] for ele in triple] for triple in val_triples]
train_triple_ids = np.array(train_triple_ids)
test_triple_ids = np.array(test_triple_ids)
val_triple_ids = np.array(val_triple_ids)
# for i in range(len(train_triples)):
#     new_triple = [id2idx[ele]]
# n_nodes = len(id2idx)
# Model and optimizer
if args.cuda:
    adj = adj.cuda()

n_nodes = adj.shape[0]

model = GCN(nfeat=args.n_feats,
            nhid=args.hidden,
            output_dim = 128,
            dropout=args.dropout, adj = adj, n_nodes=n_nodes, args=args)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()

index = torch.LongTensor(np.arange(0, n_nodes))
if args.cuda:
    index = index.cuda()

def train(epoch, e1, rel):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(e1, rel, index)
    print(output)
    exit()
    # loss_train = 
    # # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # # acc_train = accuracy(output[idx_train], labels[idx_train])
    # loss_train.backward()
    # optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
BATCHSIZE = 2048
data_size = len(train_triples)
niters = data_size // BATCHSIZE
if data_size % BATCHSIZE > 0:
    niters += 1
for epoch in range(args.epochs):
    np.random.shuffle(train_triple_ids)
    for iter in range(niters):
        train_batch = train_triple_ids[iter * BATCHSIZE: (iter + 1) * BATCHSIZE]
        train_entities = train_batch[:, 0]
        train_relations = train_batch[:, 1]
        train(epoch, train_entities, train_relations)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
