import numpy as np
import scipy.stats
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from gradient_gmm import GMM


np.random.seed(42)
torch.manual_seed(42)


N = 100000
Ns = (np.array([0.1, 0.2, 0.3, 0.4]) * N).astype(int)

data_1 = scipy.stats.multivariate_normal(mean=[1, 1], cov=[[0.1, 0], [0, 0.1]]).rvs(Ns[0])
data_2 = scipy.stats.multivariate_normal(mean=[-1, 1], cov=[[0.1, -0.09], [-0.09, 0.1]]).rvs(Ns[1])
data_3 = scipy.stats.multivariate_normal(mean=[1, -1], cov=[[0.1, 0.09], [0.09, 0.1]]).rvs(Ns[2])
data_4 = scipy.stats.multivariate_normal(mean=[-1, -1], cov=[[0.15, -0.1], [-0.1, 0.15]]).rvs(Ns[3])

data = np.vstack([data_1, data_2, data_3, data_4])

# prepare for pytorch
data_torch = torch.from_numpy(data)

fig = plt.figure(figsize=(3.75, 3.75))
plt.hist2d(data[:, 0], data[:, 1], bins=100, cmap="turbo", range=[[-3, 3], [-3, 3]])
plt.savefig("density_estimation_pngs/true_empirical_density.png", dpi=200, bbox_inches='tight')


gmm = GMM(D=2, K=4, cov_style="full")
opt = optim.Adam(gmm.parameters(), lr=3e-2)

mean_neg_log_liks = []
for i in range(1, 1000+1):
    mean_neg_log_lik = - gmm.log_prob(data_torch).mean()
    mean_neg_log_liks.append(float(mean_neg_log_lik))
    if i % 100 == 0:
        print(i, float(mean_neg_log_lik))
    opt.zero_grad()
    mean_neg_log_lik.backward()
    opt.step()

fig = plt.figure(figsize=(3.75, 3.75))
plt.plot(mean_neg_log_liks)
plt.xlabel("Number of Iterations")
plt.xlabel("Exact Negative Log-lik.")
plt.savefig("density_estimation_pngs/learning_curve.png", dpi=200, bbox_inches='tight')


with torch.no_grad():
    samples = gmm.sample(N=100000).numpy()

fig = plt.figure(figsize=(3.75, 3.75))
plt.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap="turbo", range=[[-3, 3], [-3, 3]])
plt.savefig("density_estimation_pngs/learned_empirical_density.png", dpi=200, bbox_inches='tight')
