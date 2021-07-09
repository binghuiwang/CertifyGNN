import torch
from scipy.stats import norm, binom_test
from scipy.special import comb
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

from torch.distributions.bernoulli import Bernoulli

from utils import normalize, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp

#SELF.CUDA = False
#SELF.CUDA = True

BASE = 100

class Smooth_Ber(object):

    """A smoothed classifier g """

    # to abstain, Smooth_Ber returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, dim: int, prob: float, adj: torch.tensor, fea: torch.tensor, cuda: bool):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param prob: the probability binary vector keeps the original value 
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.dim = dim
        self.prob = prob
        self.adj = adj
        self.fea = fea 
        self.cuda = cuda
        if self.cuda:
            self.m = Bernoulli(torch.tensor([self.prob]).cuda())
        else:
            self.m = Bernoulli(torch.tensor([self.prob]))


    def certify_K(self, K: int):

        pr = 0
        p = 0
        sorted_ratio = sort_ratio(K)
        shape = list(self.adj.size())

        for ele in sorted_ratio:
            u = ele[1] # 1
            v = ele[2] # 9
            p_orig, p_pertub = cal_prob(u, v)
            
            p_orig = p_orig * cal_L(K, u, v)
            p_pertub = p_pertub * cal_L(K, u, v)

            if pr + p_pertub < BASE/2 * np.power(BASE, shape[0]-1):
                pr += p_orig
                p += p_orig
            else:
                p += p_orig * (BASE/2 * np.power(BASE, shape[0]-1) - pr) /  p_pertub
                return float(p) / np.power(BASE, shape[0])


    def cal_prob(self, u: int, v: int):
        
        shape = list(self.adj.size())
        p_orig = np.power(int(self.prob * BASE), shape[0]-u) * np.power(int((1-self.prob) * BASE), u)
        p_pertub = np.power(int(self.prob * BASE), shape[0]-v) * np.power(int((1-self.prob) * BASE), v)
        return p_orig, p_pertub


    def sort_ratio(self, K: int):
        
        ratio_list = list()
        for u in range(K+1):
            for v in list(reversed(range(u, K+1))):
                if u + v >= K and np.mod(u + v - K, 2) == 0:
                    ratio_list.append((v-u,u,v))
        sorted_ratio = sorted(ratio_list, key=lambda tup: tup[0], reverse=True)
        return sorted_ratio


    def cal_L(self, K: int, u: int, v: int):
        
        shape = list(self.adj.size())
        i = int((u + v - K) / 2)
        return comb(shape[0]-K, i) * comb(K, u-i)



    def certify_Ber(self, x: int, n0: int, n: int, alpha: float, batch_size: int):
        """
        p(0->0) = p(1->1) = prob
        p(0->1) = p(1->0) = 1 - prob
        """
        
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some bernoulli noise.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within some bernoulli noise around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise_ber(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise_ber(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        # print('ppf:', norm.ppf(pABar))

        if pABar < 0.5:
            return Smooth_Ber.ABSTAIN, 0.0
        else:
            return cAHat, pABar
            


    def predict_Ber(self, x: int, n: int, alpha: float, batch_size: int):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """

        self.base_classifier.eval()
        counts = self._sample_noise_ber(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        #print(top2)

        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        #print(count1, count2)

        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth_Ber.ABSTAIN
        else:
            return top2[0]


    def _sample_noise_ber(self, idx: int, num: int, batch_size: int):
        """ Sample the base classifier's prediction under bernoulli noisy of input x's adj vector.
        :param idx: the input index
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        
        if self.cuda:
            adj = self.adj.to_dense().int().clone().detach().cuda()
            adj_noise = adj.clone().detach().cuda()
        else:
            adj = self.adj.to_dense().int().clone().detach()
            adj_noise = adj.clone().detach()

        shape = list(adj.size())
        print(shape)

        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            
            for _ in range(num):
                
                mask = self.m.sample(adj[idx].shape).squeeze(-1).int()
                
                if self.cuda:
                    rand_inputs = torch.randint_like(adj[idx], low=0, high=2, device='cuda').squeeze().int()
                else:
                    rand_inputs = torch.randint_like(adj[idx], low=0, high=2).squeeze().int()

                adj_noise[idx] = adj[idx] * mask + rand_inputs * (1 - mask)
                

                #print('#nnz:', (adj_noise[idx] - adj[idx]).sum())
                adj_noise[:,idx] = adj_noise[idx]

                adj_noise_norm = normalize(adj_noise.cpu().numpy() + sp.eye(adj_noise.cpu().numpy().shape[0]))
                adj_noise_norm = sp.coo_matrix(adj_noise_norm)
                if self.cuda:
                    adj_noise_norm = sparse_mx_to_torch_sparse_tensor(adj_noise_norm).to(device='cuda')
                else:
                    adj_noise_norm = sparse_mx_to_torch_sparse_tensor(adj_noise_norm)
                    
                predictions = self.base_classifier(self.fea, adj_noise_norm).argmax(1)
                prediction = predictions[idx]
                counts[prediction.cpu().numpy()] += 1
                
            
            print(counts)
            return counts
            

    def _count_arr(self, arr: np.ndarray, length: int):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float):
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]