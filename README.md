# SIM-Shapely

Explainable artificial intelligence (XAI) is critical for ensuring trust and accountability in machine learning (ML), particularly in high-stakes domains such as healthcare and finance. Shapley value (SV)-based methods provide a principled framework for feature attribution in complex ML models but suffer from high computational costs, limiting their scalability to high-dimensional problems. We propose Stochastic Iterative Momentum for Shapley Value Approximation (**SIM-Shapley**), a stable and efficient SV approximation method inspired by stochastic optimization. We establish theoretical guarantees on bias and variance, prove linear $Q$-convergence, and demonstrate improved empirical stability on real-world datasets. 
Specifically, in our numerical experiments, SIM-Shapley reduces computation time by up to 85\% compared to state-of-the-art baselines, while maintaining comparable feature attribution quality.
Beyond feature attribution, our stochastic mini-batch iterative framework extends naturally to a broader class of sample average approximation problems, offering a new avenue for improving computational efficiency while guaranteeing stability.


