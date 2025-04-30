# Simulation 1 description

In this simulation we consider good separation of the layer distributions, but decreasing feature separation. Specifically, we set:
$$
\boldsymbol{\gamma}_1 = (1,0,0), \qquad \boldsymbol{\gamma}_2 = (0,1,0), \qquad \boldsymbol{\gamma}_3 = (0,0,1),
$$
and sample their features from multivariate Gaussians with identity covariance matrix and means:
$$
\boldsymbol{\mu}_1 = \alpha (1,1,1),\qquad \boldsymbol{\mu}_2 = (0,0,0), \qquad \boldsymbol{\mu}_3 = \alpha (-1,-1,-1),
$$
where we allow $\alpha$ to vary over the set $\{5, 5/2, 2, 3/2, 1, 1/2\}$. We run 50 simulations for each value of $\alpha$. 