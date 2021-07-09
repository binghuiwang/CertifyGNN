from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import datetime
from core_Ber import Smooth_Ber

import torch
import torch.nn.functional as F
import torch.optim as optim

import scipy.sparse as sp
from utils import load_data, load_data_new, accuracy, normalize, sparse_mx_to_torch_sparse_tensor
from models import GCN

import os

# Training settings
#DATASETS = ["cora", "citeseer", "pubmed"]

parser = argparse.ArgumentParser()
#parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument("--dataset", type=str, default="cora", help="which dataset")
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument("--batch", type=int, default=10000, help="batch size")
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--prob', default=0.8, type=float,
                    help="probability to keep the status for each binary entry")
parser.add_argument('--beta', default=0.0, type=float,
                    help="propagation factor")

parser.add_argument("--predictfile", type=str, help="output prediction file")
parser.add_argument("--certifyfile", type=str, help="output certified file")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")


args = parser.parse_args()
print(args.cuda)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    torch.cuda.manual_seed(args.seed)


##### Note: In the original GCN, the adjacent matrix is degree normalized. 
##### Here, to certify the number of edges, we use the binary adjacent matrix 

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = \
#     load_data(path="../data/cora/", dataset=args.dataset, alpha=args.beta, n_iter=4)


adj, features, labels, idx_train, idx_val, idx_test = \
    load_data_new(dataset=args.dataset, alpha=args.beta, n_iter=4)
adj_norm = normalize(adj + sp.eye(adj.shape[0]))

adj = sparse_mx_to_torch_sparse_tensor(adj)
adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

print(features.shape)

if args.cuda:
    #model.cuda()
    features = features.cuda()
    adj_norm = adj_norm.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
else:
    pass
    

# GCN
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()


def get_num_classes_dim(dataset):
    """Return the number of classes in the dataset. """
    if dataset == "cora":
        num_class, dim = 7, 1433
    if dataset == "citeseer":
        num_class, dim = 6, 3703
    elif dataset == "pubmed":
        num_class, dim = 3, 500

    return num_class, dim

    
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj_norm)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj_norm)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj_norm)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def certify():
    num_class, dim = get_num_classes_dim(args.dataset)
    # create the smoothed classifier g
    smoothed_classifier = Smooth_Ber(model, num_class, dim, args.prob, adj, features, args.cuda)

    # prepare output file
    f = open(args.certifyfile, 'w')
    print("idx\tlabel\tpredict\tpABar\tcorrect\ttime", file=f, flush=True)

    cnt = 0
    cnt_certify = 0
    
    for i in idx_test:
    #for i in idx_test[:10]:
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        before_time = time.time()
        # make the prediction
        prediction, pABar = smoothed_classifier.certify_Ber(i, args.N0, args.N, args.alpha, args.batch)
        #print(prediction, labels[i])
        after_time = time.time()
        correct = int(prediction == labels[i])

        cnt += 1
        cnt_certify += correct

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}\t{}".format(i, labels[i], prediction, pABar, correct, time_elapsed), file=f, flush=True)

    f.close()

    #print("certify acc:", float(cnt_certify) / cnt)



if __name__ == "__main__":

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):

        train(epoch)
        
        torch.save({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.prob.'+str(args.prob)))


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    
    #### Testing
    #test()


    ## Certify
    certify()

