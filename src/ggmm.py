import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
import torch.distributions as tdist


class GMM(nn.Module):
    
    def __init__(self, D, K, cov_style="full", μs_init_min_and_max=None):
        
        super().__init__()
        
        self.K = K
        self.D = D
        self.cov_style = cov_style
        
        self.pre_π = nn.Parameter(torch.zeros(self.K, ))
        
        if μs_init_min_and_max is None:
            self.μs = nn.Parameter((torch.rand(self.K, self.D) - 0.5) * 2)
        else:
            μs_init_range = μs_init_min_and_max[1] - μs_init_min_and_max[0]
            self.μs = nn.Parameter(
                torch.rand(self.K, self.D) * μs_init_range.reshape(1, self.D) + μs_init_min_and_max[0].reshape(1, self.D)
            )
        
        if self.cov_style == "full":
            Σs_init = torch.eye(D).reshape(1, D, D).repeat(self.K, 1, 1)
            self.pre_Σs = nn.Parameter(linalg.cholesky(Σs_init, upper=True))
        elif self.cov_style == "diag":
            self.pre_Σs = torch.nn.Parameter(torch.rand(self.K, self.D))
        else:
            raise ValueError(f"{cov_style} is an invalid covariance style.")
            
    @property
    def π(self):
        return torch.softmax(self.pre_π, dim=0)
    
    @property
    def categorical(self):
        return tdist.Categorical(logits=self.pre_π)
    
    @property
    def Σs(self):
        if self.cov_style == "full":
            Ls = torch.triu(self.pre_Σs)
            return torch.bmm(Ls.transpose(1, 2), Ls)
        elif self.cov_style == "diag":
            # (K, D, 1) * (1, D, D) =(broadcast)=> (K, D, D) * (K, D, D) =(elementwise)=> (K, D, D)
            return (self.pre_Σs ** 2).unsqueeze(-1) * torch.eye(self.D).unsqueeze(0)
    
    @property
    def gaussians(self):
        return tdist.MultivariateNormal(self.μs, self.Σs)
        
    def log_prob(self, samples):
        
        # this function just implements the log probability of the gmm model in a numerically stable way
        
        N = samples.shape[0]
        log_π = F.log_softmax(self.pre_π, dim=0).reshape(1, -1)  # (1, K)
        log_probs_under_each_gaussian = self.gaussians.log_prob(samples.view(N, 1, self.D))  # (N, K)
        log_probs = torch.logsumexp(log_π + log_probs_under_each_gaussian, dim=1)
        
        return log_probs
    
    def sample(self, N):
        
        indices = self.categorical.sample((N, ))  # (N,)
        samples_from_each_gaussian = self.gaussians.rsample(sample_shape=(N, ))  # (n, K, D)
        
        # from pytorch doc for gather: out[i][j][k] = input[i][index[i][j][k]][k]
        # we know that we want samples to be of shape (N, D), so out should have shape (N, 1, D)
        
        samples = samples_from_each_gaussian.gather(
            dim=1,  # we will be indexing through the 1-th dimension (K)
            index=indices.view(-1, 1, 1).repeat(1, 1, self.D)
        ).squeeze()  # (N, D)
        
        return samples
    
    def compute_kl(self, log_unnorm_p, N_per_gaussian):
        
        # by pass reparameterizing the categorical using marginalization
        # (this has much lower variance)
        
        samples = self.gaussians.rsample((N_per_gaussian, )).reshape(N_per_gaussian * self.K, self.D)
        kl = (self.π.repeat(N_per_gaussian) * (self.log_prob(samples) - log_unnorm_p(samples))).sum() / N_per_gaussian
        
        return kl