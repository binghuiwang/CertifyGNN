import numpy as np
import scipy.sparse as sp
import torch
import sys

import pickle as pkl
import networkx as nx


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_logfile(filename, text):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename, text):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data_new(dataset, alpha=0.0, n_iter=4): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    """
    Loads input data from gcn/data directory

    ind.dataset.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset.ally => the labels for instances in ind.dataset.allx as numpy.ndarray object;
    ind.dataset.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    #print(graph)
    print(allx.shape,tx.shape)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #print(features.shape)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #print(labels.shape)

    idx_test = test_idx_range.tolist()
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    #print('#test:', len(idx_test), )


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if alpha == 0.0:
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = normalize_prop(features, adj, alpha, n_iter, normFea=False)
        features = torch.FloatTensor(np.array(features))

    #adj = normalize(adj + sp.eye(adj.shape[0]))

    # label_train = y.argmax(axis=1)
    # idx_train = []
    # class_list = [[] for _ in range(y.shape[1])]
    # for cls in range(y.shape[1]):
    #     class_list[cls] = np.where(label_train == cls)[0]
    #     # print(class_list[cls])
    #     #print(len(class_list[cls]))
    #     idx_train.extend(class_list[cls][:int(len(class_list[cls]))])
    #     # print idx_train
    # idx_val = range(len(y), len(y)+500)


    #labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(np.argmax(labels,1))

    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(path="../data/cora/", dataset="../data/cora", alpha=0.0, n_iter=4):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if dataset == "cora":

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

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if alpha == 0.0:
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = normalize_prop(features, adj, alpha, n_iter, normFea=False)
        features = torch.FloatTensor(np.array(features))

    adj = normalize(adj + sp.eye(adj.shape[0]))

    if dataset == 'cora':
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    if dataset == 'citeseer':
        pass

    if dataset == 'pubmed':
        pass

    #features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    #r_mat_inv = sp.diags(r_inv)
    r_mat_inv = sp.diags(r_inv,0)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_prop(mx, adj, alpha, n_iter, normFea=False):

    if normFea:
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        # print(r_inv.shape)
        # print(r_inv)
        r_mat_inv = sp.diags(r_inv,0)
        #r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
    else:
        mx = mx.todense()

    """Feature propagation via Normalized Laplacian"""
    S = normalize_adj(adj)
    F = alpha * S.dot(mx) + (1-alpha) * mx
    for _ in range(n_iter):
        F = alpha * S.dot(F) + (1-alpha) * mx
    return F


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



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
