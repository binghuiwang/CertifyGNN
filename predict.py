import argparse
#import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime
import os

DATASETS = ["cora", "citeseer", "pubmed"]


parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
#parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()



if __name__ == "__main__":

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = \
        load_data(path="../data/cora/", dataset=args.dataset, alpha=args.alpha, n_iter=4)
    print(features.shape)


    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    if args.dataset == "cora":
        dim = 1433
    elif args.dataset == "citeseer":
        dim = 3703
    elif args.dataset == "pubmed":
        dim = 500

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, args.dim)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)

    cnt = 0
    cnt_pred = 0

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    
    model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

    model.eval()
    output = model(features, adj)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(alpha=args.alpha)

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        before_time = time()

        # make the prediction
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)

        after_time = time()
        correct = int(prediction == label)

        cnt += 1
        cnt_pred += correct

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), file=f, flush=True)

    f.close()

    print("test acc:", float(cnt_pred) / cnt)
