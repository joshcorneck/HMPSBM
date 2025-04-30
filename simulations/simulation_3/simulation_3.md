# Simulation 3 description

In this simulation, we demonstrate that our inference procedure can correctly recover the number of global and layer-level groups when the truncations $M_w$ and $M_z$ are larger than the true simulation values. We consider two global groups, with 300 and 200 nodes in group 1 and 2, respectively. We set the probability vectors over the three layer-level groups to be:
$$
    \boldsymbol{\gamma}_1 = (0.8, 0.1, 0.1), \quad \boldsymbol{\gamma}_2 = (0, 0.5, 0.5),
$$
and sample their features from multivariate normals with identity covariance matrices and mean vectors
$$
    \boldsymbol{\mu}_1 = (3/2,3/2,3/2), \quad \boldsymbol{\mu}_2 = (-3/2,-3/2,-3/2).
$$
We set the probability matrix $\boldsymbol{\rho}$ to be
$$
    \boldsymbol{\rho} = 
    \begin{pmatrix}
        0.8 & 0.5 & 0.2 \\
        0.4 & 0.7 & 0.05 \\
        0.2 & 0.01 & 0.6
    \end{pmatrix},
$$
and run the CAVI inference procedure for 5 iterations. We run the procedure 50 times with $M_w = 2$ and $M_z=3$, and again with $M_w=M_z=5$. 