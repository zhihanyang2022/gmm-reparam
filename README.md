# Fitting mixtures of full-rank Gaussians using gradient

Local minima is a known problem for latent variable models. Common solutions include random starts and good initialization (e.g., K-means). In 2D, this problem can be by-passed by having a mixture of many Gaussians, as I have done here. This might be a good motivation for putting a sparsity prior over the mixture weights, as done in [Bayesian mixture of Gaussians](https://github.com/zhihanyang2022/bayesian-mixture-of-gaussians), though such a model is too complicated. For higher dimensions, having many Gaussians might not be a good strategy due to the curse of dimensionality (i.e., how many is enough?), though my code supports arbitrary dimensionality and would run fine.

## Example 1: density estimation given a dataset

Legend:
- 1st image: true (empirical) density (from data)
- 2nd image: learned (empirical) density
- 3rd image: learning curve

<p align="middle">
  <img src="examples/density_estimation_pngs/true_empirical_density.png" width="30%" />
  <img src="examples/density_estimation_pngs/learned_empirical_density.png" width="30%" /> 
  <img src="examples/density_estimation_pngs/learning_curve.png" width="30%" />
</p>

## Example 2: variational inference given the log of an unnormalized density 

Legend:

- 1st image: true (unnormalized) density
- 2nd image: learned (empirical) density with arrows showing the initial and final positions of Gaussian means
- 3rd image: learned mixture weights
- 4th image: learning curve

### Potential function U2

<p align="middle">
  <img src="examples/variational_inference_pngs/U1/true_unnormalized_density.png" width="20%" />
  <img src="examples/variational_inference_pngs/U1/learned_empirical_density.png" width="20%" /> 
  <img src="examples/variational_inference_pngs/U1/mixture_weights.png" width="20%" />
  <img src="examples/variational_inference_pngs/U1/learning_curve.png" width="20%" />
</p>

### Potential function U3

<p align="middle">
  <img src="examples/variational_inference_pngs/U2/true_unnormalized_density.png" width="20%" />
  <img src="examples/variational_inference_pngs/U2/learned_empirical_density.png" width="20%" /> 
  <img src="examples/variational_inference_pngs/U2/mixture_weights.png" width="20%" />
  <img src="examples/variational_inference_pngs/U2/learning_curve.png" width="20%" />
</p>

### Potential function U3

<p align="middle">
  <img src="examples/variational_inference_pngs/U3/true_unnormalized_density.png" width="20%" />
  <img src="examples/variational_inference_pngs/U3/learned_empirical_density.png" width="20%" /> 
  <img src="examples/variational_inference_pngs/U3/mixture_weights.png" width="20%" />
  <img src="examples/variational_inference_pngs/U3/learning_curve.png" width="20%" />
</p>

### Potential function U4

<p align="middle">
  <img src="examples/variational_inference_pngs/U4/true_unnormalized_density.png" width="20%" />
  <img src="examples/variational_inference_pngs/U4/learned_empirical_density.png" width="20%" /> 
  <img src="examples/variational_inference_pngs/U4/mixture_weights.png" width="20%" />
  <img src="examples/variational_inference_pngs/U4/learning_curve.png" width="20%" />
</p>

### Potential function U8

<p align="middle">
  <img src="examples/variational_inference_pngs/U8/true_unnormalized_density.png" width="20%" />
  <img src="examples/variational_inference_pngs/U8/learned_empirical_density.png" width="20%" /> 
  <img src="examples/variational_inference_pngs/U8/mixture_weights.png" width="20%" />
  <img src="examples/variational_inference_pngs/U8/learning_curve.png" width="20%" />
</p>
