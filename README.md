# SIM-Shapely

Explainable artificial intelligence (XAI) is essential for trustworthy machine learning (ML), particularly in high-stakes domains such as healthcare and finance. Shapley value (SV) methods provide a principled framework for feature attribution in complex models but incur high computational costs, limiting their scalability in high-dimensional settings.
We propose Stochastic Iterative Momentum for Shapley Value Approximation (**SIM-Shapley**), a stable and efficient SV approximation method inspired by stochastic optimization. We analyze variance theoretically, prove linear $ Q $-convergence, and demonstrate improved empirical stability and low bias in practice on real-world datasets.
In our numerical experiments, SIM-Shapley reduces computation time by up to 85% relative to state-of-the-art baselines while maintaining comparable feature attribution quality.
Beyond feature attribution, our stochastic mini-batch iterative framework extends naturally to a broader class of sample average approximation problems, offering a new avenue for improving computational efficiency with stability guarantees.

