# Implementation for preprint SIM-Shapley: A Stable and Computationally Efficient Approach to Shapley Value Approximation

Read the complete story in our latest [preprint](https://arxiv.org/abs/2505.08198).

## Abstract

Explainable artificial intelligence (XAI) is essential for trustworthy machine learning (ML), particularly in high-stakes domains such as healthcare and finance. Shapley value (SV) methods provide a principled framework for feature attribution in complex models but incur high computational costs, limiting their scalability in high-dimensional settings.
We propose Stochastic Iterative Momentum for Shapley Value Approximation (**SIM-Shapley**), a stable and efficient SV approximation method inspired by stochastic optimization. We analyze variance theoretically, prove linear $Q$-convergence, and demonstrate improved empirical stability and low bias in practice on real-world datasets.
In our numerical experiments, SIM-Shapley reduces computation time by up to 85% relative to state-of-the-art baselines while maintaining comparable feature attribution quality.
Beyond feature attribution, our stochastic mini-batch iterative framework extends naturally to a broader class of sample average approximation problems, offering a new avenue for improving computational efficiency with stability guarantees.

## External Library Description

[**sage**](https://github.com/nliulab/SIM-Shapley/tree/main/sage): the implementation of [SAGE paper](https://arxiv.org/abs/2004.00668). We add some code to record convergence process, nothing crucial changed. See original code in this [repo](https://github.com/iancovert/sage/).

[**shapreg**](https://github.com/nliulab/SIM-Shapley/tree/main/shapreg): the impelmentation of [SHAP paper](https://arxiv.org/abs/2012.01536). We add some code to record convergence process, nothing crucial changed. See original code in this [repo](https://github.com/iancovert/shapley-regression).

## Demo

We offer several demos to display how to run our algo to estimate Shapely values. The Pearson's Coeffecient Correlation and Wasserstein Distance are used to measure the distribution differences.

Addtionally, [credit](https://github.com/nliulab/SIM-Shapley/blob/main/demo/credit.ipynb) offers a script to visualize convergence process, and [bike](https://github.com/nliulab/SIM-Shapley/blob/main/demo/bike.ipynb) illustrates the consistency of various estimation methods using figures.
