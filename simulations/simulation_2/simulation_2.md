# Simulation 2 description

In this simulation we consider good feature separation, but decreasing separation of the layer groups. Specifically, we set:
$$
\boldsymbol{\gamma}_1 = (1 - 2\alpha,\alpha,\alpha), \qquad \boldsymbol{\gamma}_2 = (\alpha,1-2\alpha,\alpha_, \qquad \boldsymbol{\gamma}_3 = (\alpha,\alpha,1-2\alpha),
$$
where we allow $\alpha$ to vary over the set $\{0.05,  0.1, 0.15, 0.2, 0.25, 0.33\}$. The features are sampled from multivariate Gaussians with identity covariance matrix and means:
$$
\boldsymbol{\mu}_1 = \alpha (1,1,1),\qquad \boldsymbol{\mu}_2 = \alpha (0,0,0), \qquad \boldsymbol{\mu}_3 = \alpha (-1,-1,-1).
$$
These choices of mean selects distributions for the features that will samples that cover overlapping regions with positive probability. We run 50 simulations for each value of $\alpha$. 