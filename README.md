# Graphical Models: Automatic Differentiation Variational Inference (ADVI)

## Overview

This repository provides an implementation of the **Automatic Differentiation Variational Inference (ADVI)** algorithm, applied to the spatial-temporal analysis of the Porto taxi trajectories dataset.

This project is directly based on the work of Kucukelbir et al. (2016). While traditional Variational Inference (VI) requires model-specific analytical derivations and is generally restricted to conditionally conjugate models, ADVI lifts this restriction by:

1. Transforming the support of the latent variables $L$ via a differentiable bijection $T : \text{supp}(L) \rightarrow \mathbb{R}^K$ into an unconstrained continuous space.
2. Formulating a Gaussian variational approximation in this unconstrained real coordinate space.
3. Leveraging elliptical standardization (the *reparameterization trick*) and Monte Carlo integration to evaluate expectations.
4. Optimizing the variational parameters using stochastic gradient ascent, making the inference process fully automatable.

**Reference:** Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2016). *Automatic Differentiation Variational Inference*. arXiv:1603.00788.

## Objective

The main objective of this notebook is to cluster 1.7 million taxi trajectories using a **non-conjugate Gaussian Mixture Model (GMM)** implemented .

Because exact Bayesian inference is intractable in high dimensions (computing $\int P(L, O) dL$ is impossible) and standard Markov Chain Monte Carlo (MCMC) methods are computationally prohibitive for such a large volume of data, we leverage the ADVI framework to scale the inference efficiently.

## Optimization Metric: ELBO

To bypass the computation of the exact posterior distribution $P(L|O)$, we seek an approximate distribution $Q_\theta(L)$ that minimizes the Kullback-Leibler divergence $D_{KL}(Q_\theta(L) || P(L|O))$.

Mathematically, minimizing this divergence is strictly equivalent to **maximizing the Evidence Lower Bound (ELBO)**, denoted as $\mathcal{L}(\theta)$. This is the objective function implemented and optimized in this project:

$$\mathcal{L}(\theta) = \mathbb{E}*{Q*\theta} [\log P(O, L)] - \mathbb{E}*{Q*\theta} [\log Q_\theta(L)]$$

Where:

- $O$ represents the set of observed variables (the trajectories).
- $L$ represents the set of latent variables of the mixture model.
- $Q_\theta$ is the variational distribution parameterized by $\theta$.

The ELBO achieves an optimal trade-off between the expected log-likelihood of the data (model fit) and the entropy of the variational distribution (regularization).