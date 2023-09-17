import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import math

from gradient_gmm import GMM
from potential_funcs import pick_potential_func

# ==================================================
# Contour plot of potential function of choice
# ==================================================

parser = argparse.ArgumentParser()
parser.add_argument("index", help="Index of some potential function")
parser.add_argument("--seed", default=42)
args = parser.parse_args()
config = vars(args)

index = int(config["index"])
seed = int(config["seed"])

torch.manual_seed(seed)
U, K = pick_potential_func(index)

folder = f"variational_inference_pngs/U{index}"
os.makedirs(folder, exist_ok=True)

# ==================================================
# Contour plot of potential function of choice
# ==================================================

fig = plt.figure(figsize=(3.75, 3.75))

xs = torch.linspace(-6, 6, 100)
xxs, yys = torch.meshgrid(xs, xs)
xxs2, yys2 = xxs.reshape(-1, 1), yys.reshape(-1, 1)
unnorm_p = torch.exp(- U(torch.hstack([xxs2, yys2])))

plt.contourf(xxs.numpy(), yys.numpy(), unnorm_p.reshape(100, 100).numpy(), levels=100, cmap="turbo")
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"{folder}/true_unnormalized_density.png", dpi=200, bbox_inches='tight')

# ==================================================
# Code for variational inference
# ==================================================

q = GMM(
    D=2,
    K=K,
    μs_init_min_and_max=torch.tensor([
        [-6., -6.],
        [6., 6.]]
    )
)
opt = optim.Adam(q.parameters(), lr=1e-2)

μs_init = q.μs.detach().numpy().copy()
N = 1000
N_per_gaussian = math.ceil(N / K)

sample_kls = []

for i in range(1, 10001):

    sample_kl = q.compute_kl(log_unnorm_p=lambda x: -U(x), N_per_gaussian=N_per_gaussian)

    sample_kls.append(float(sample_kl))

    opt.zero_grad()
    sample_kl.backward()
    opt.step()

    if i % 1000 == 0:
        print("Iteration:", i, " KL:", sum(sample_kls[-30:]) / 30)

# ==================================================
# Learning curve
# ==================================================

fig = plt.figure(figsize=(3.75, 3.75))
plt.plot(np.arange(1, len(sample_kls) + 1), sample_kls, linewidth=1, color="black", alpha=0.3)
plt.scatter(np.arange(1, len(sample_kls) + 1), sample_kls, s=2, color="black")

plt.ylabel(r"Stochastic Estimate of $\mathbb{KL}(q \vert \vert \tilde{p})$")
plt.xlabel("Number of iterations")
plt.gca().spines[['right', 'top']].set_visible(False)

plt.grid()
plt.xscale("log")
y_range = max(sample_kls) - min(sample_kls)
plt.ylim(min(sample_kls) - 1 / 20 * y_range, max(sample_kls) + 1 / 20 * y_range)

plt.savefig(f"{folder}/learning_curve.png", dpi=200, bbox_inches='tight')

# ==================================================
# Contour plot of learned density (empirical)
# ==================================================

fig = plt.figure(figsize=(3.75, 3.75))

with torch.no_grad():
    q_samples = q.sample(N=int(1e6)).numpy()
    μs = q.μs.numpy()

plt.hist2d(q_samples[:, 0], q_samples[:, 1], bins=100, cmap="turbo", range=[[-6, 6], [-6, 6]])
for i, μ in enumerate(μs):
    plt.arrow(
        x=μs_init[i][0],
        y=μs_init[i][1],
        dx=μ[0] - μs_init[i][0],
        dy=μ[1] - μs_init[i][1],
        color="white",
        head_width=0.15,
        length_includes_head=True
    )

plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig(f"{folder}/learned_empirical_density.png", dpi=200, bbox_inches='tight')

# ==================================================
# Bar-plot of mixture weights
# ==================================================

fig = plt.figure(figsize=(3.75, 3.75))
plt.bar(np.arange(K), q.π.detach().numpy(), color="black")
plt.axhline(1 / K, linestyle="--", color="black", label="1/K")

plt.xlabel("Component Index (1 to K)")
plt.ylabel("Mixing Probability")
plt.legend()
plt.gca().spines[['right', 'top']].set_visible(False)

plt.xticks([])

plt.savefig(f"{folder}/mixture_weights.png", dpi=200, bbox_inches='tight')
