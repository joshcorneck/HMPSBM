# Simulation 3 description

In this simulation we consider good feature separation, and moderate separation between the behaviour of global groups in each layer. However, we select a $\rho$ matrix with moderate separation and also for the number of global and layer-level groups to be inferred. We set $K=2$ and $K_1=\dots=K_{10}=3$, with 
$$
\rho = 
\begin{pmatrix}
0.8 & 0.5 & 0.3 \\
0.7 & 0.7 & 0.4 \\
0.3 & 0.35 & 0.5
\end{pmatrix}.
$$
We set 
$$
\boldsymbol{\gamma}_1 = (0.7, 0.2, 0.1), \qquad \boldsymbol{\gamma}_2  = (0, 0.2, 0.8),
$$
with 
$$
\boldsymbol{\mu}_1 = (10, 10, 10), \qquad \boldsymbol{\mu}_2 = (-10,-10,-10).
$$
For inference, we set $M_w = M_z = 5$. We run this for 1, 2, 5 and 10 i.i.d. repetitions on each edge. 