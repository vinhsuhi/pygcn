import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_old(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def create_adj(train_triples, entity_set, relation_set):
    id2idx = dict()
    # relation_id2idx = dict()
    count = 0
    for ele in entity_set:
        id2idx[ele] = count 
        count += 1
    for ele in relation_set:
        id2idx[ele] = count 
        count += 1
    
    entity_count_dict = dict()
    relation_count_dict = dict()
    entity_entity_count_dict = dict()
    entity_relation_count_dict = dict()

    def update_count(ele, dictt, id2idx):
        if id2idx[ele] not in dictt:
            dictt[id2idx[ele]] = 1
        else:
            dictt[id2idx[ele]] += 1
    
    def update_count_special(ele1, ele2, dictt, id2idx):
        source = str(id2idx[ele1])
        target = str(id2idx[ele2])
        key = "{}_{}".format(source, target)
        if key not in dictt:
            dictt[key] = 1
        else:
            dictt[key] += 1

    for i in range(len(train_triples)):
        this_triple = train_triples[i]
        entity1, relation, entity2 = this_triple[0], this_triple[1], this_triple[2]
        update_count(entity1, entity_count_dict, id2idx)
        update_count(entity2, entity_count_dict, id2idx)
        update_count(relation, relation_count_dict, id2idx)
        update_count_special(entity1, entity2, entity_entity_count_dict, id2idx)
        update_count_special(entity1, relation, entity_relation_count_dict, id2idx)
        update_count_special(entity2, relation, entity_relation_count_dict, id2idx)
    
    edges = []
    values = []
    for ele in train_triples:
        entity1, relation, entity2 = ele[0], ele[1], ele[2]
        edges.append([id2idx[entity1], id2idx[entity2]])
        edges.append([id2idx[entity1], id2idx[relation]])
        edges.append([id2idx[relation], id2idx[entity2]])
        e1_count = entity_count_dict[id2idx[entity1]] / len(train_triples)
        e2_count = entity_count_dict[id2idx[entity2]] / len(train_triples)
        r_count = relation_count_dict[id2idx[relation]] / len(train_triples)
        e1e2_count = entity_entity_count_dict["{}_{}".format(id2idx[entity1], id2idx[entity2])] / len(train_triples)
        e1r_count = entity_relation_count_dict["{}_{}".format(id2idx[entity1], id2idx[relation])] / len(train_triples)
        e2r_count = entity_relation_count_dict["{}_{}".format(id2idx[entity2], id2idx[relation])] / len(train_triples)
        values.append(e1e2_count / (e1_count * e2_count))
        values.append(e1r_count / (e1_count * r_count))
        values.append(e2r_count / (e2_count * r_count))

    edges = np.array(edges)
    values = np.array(values)
    adj = sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(max(list(id2idx.values())) + 1,  max(list(id2idx.values())) + 1))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, id2idx


def load_data(path_train, path_test, path_val):
    entity_set = set()
    relation_set = set()
    train_triples = []
    val_triples = []
    test_triples = []
    with open(path_train, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            if data_line[0] not in entity_set:
                entity_set.add(data_line[0])
            if data_line[-1] not in entity_set:
                entity_set.add(data_line[-1])
            if data_line[1] not in relation_set:
                relation_set.add(data_line[1])
            train_triples.append(data_line)
    file.close()

    with open(path_val, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            if data_line[0] not in entity_set:
                entity_set.add(data_line[0])
            if data_line[-1] not in entity_set:
                entity_set.add(data_line[-1])
            if data_line[1] not in relation_set:
                relation_set.add(data_line[1])
            val_triples.append(data_line)
    file.close()

    with open(path_test, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            if data_line[0] not in entity_set:
                entity_set.add(data_line[0])
            if data_line[-1] not in entity_set:
                entity_set.add(data_line[-1])
            if data_line[1] not in relation_set:
                relation_set.add(data_line[1])
            test_triples.append(data_line)
    file.close()

    return entity_set, relation_set, train_triples, val_triples, test_triples
            

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
