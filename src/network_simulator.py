import numpy as np

from scipy.stats import norm
from scipy.stats import invgamma

def create_orthogonal_sets(nodes_in_groups: np.array, 
                           perturbation_scale: float,
                           feat_dim: int):
    """
    Method for producing sets of orthogonal vectors to be used as features in 
    model simulation.
    Parameters:
        - nodes_in_groups: numpy array giving the number of nodes in
                           each of the global groups.
        - perturbation_scale: width of the uniform distribution (symmetric
                              about 0) that we use to perturb.
        - feat_dim: P, the dimension of each feature vector
    """
    num_groups = len(nodes_in_groups)
    total_nodes = int(np.sum(nodes_in_groups))
    sampled_features = np.zeros((total_nodes, feat_dim))

    base_vecs = np.zeros((num_groups, feat_dim))
    start_idx = 0
    
    for group in range(num_groups):
        # Generate the base vector for the current group
        base_mean = np.random.uniform(low=-1, high=1)
        base_vec = np.random.normal(loc=base_mean, scale=1, size=(feat_dim,))
        
        if group == 0:
            base_vecs[group, :] = base_vec
        else:
            # Make the base vector orthogonal to all previous base vectors
            for g in range(group):
                projection = (np.dot(base_vec, base_vecs[g, :]) / 
                              np.dot(base_vecs[g, :], base_vecs[g, :])) * base_vecs[g, :]
                base_vec = base_vec - projection
            base_vecs[group, :] = base_vec

        # Normalize the base vector
        base_vecs[group, :] /= np.linalg.norm(base_vecs[group, :])

        # Generate the feature vectors for the current group
        num_nodes = nodes_in_groups[group]
        for sample in range(num_nodes):
            sampled_features[start_idx + sample, :] = (
                base_vecs[group, :] + np.random.uniform(low=-perturbation_scale/2,
                                                        high=perturbation_scale/2,
                                                        size=(feat_dim, ))
            )
        start_idx += num_nodes  # Move to the next group starting index

    return sampled_features

class MultiplexSimulator:

    def __init__(self, num_nodes: int, num_layers: int, num_glob_groups: int,
                 max_num_layer_groups: np.array, features: np.array,
                 specify: bool=True, rho_matrix: np.array=None, 
                 layer_groups: np.array=None, zeta: np.array=None, 
                 phi: np.array=None, gamma: np.array=None,
                 num_adj_samps: int=1):
        """
        A class for simulating a multiplex network from the model.
        Parameters: 
            - num_nodes: N, number of nodes in the network.
            - num_layers: L, number of layers in the network.
            - num_glob_groups: K, the number of latent global groups.
            - max_num_layer_groups: max_l(K_l).
            - features: x, numpy array of node features.
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_glob_groups = num_glob_groups
        self.max_num_layer_groups = max_num_layer_groups
        self.num_adj_samps = num_adj_samps

        self.features = features
        self.P = features.shape[1]
        
        self.specify = specify
        
        # Checks
        if self.specify:
            if layer_groups is None:
                raise ValueError("You must suuply layer_groups if specify = True.")
            elif layer_groups.shape != (num_layers, num_nodes):
                raise ValueError("The shape of layer_groups is incorrect.")
            else:
                self.layer_groups = layer_groups
            if rho_matrix is None:
                raise ValueError("""You must supply a connectivty matrix
                                 if specify = True.""")
        if rho_matrix is not None:
            # M = max(num_glob_groups, max_num_layer_groups)
            M = max_num_layer_groups
            if rho_matrix.shape != (M, M):
                raise ValueError("")
        if zeta is not None:
            if zeta.shape != (num_glob_groups, max_num_layer_groups):
                raise ValueError("")
        if phi is not None:
            if phi.shape != (num_glob_groups, self.P):
                raise ValueError("")
        if gamma is not None:
            if gamma.shape != (num_glob_groups, max_num_layer_groups):
                raise ValueError("")
        
        self.rho_matrix = rho_matrix
        self.zeta = zeta
        self.phi = phi
        self.gamma = gamma

        # self.sigma2 = np.tile([0.005], self.num_glob_groups)

    def _sample_pi(self, xi_0):
        """
        """
        self.pi = np.ones((self.max_num_layer_groups,))
        pi_prime = np.random.beta(a=1, b=xi_0, size=((self.max_num_layer_groups, )))
        self.pi[1:] = np.cumprod(1 - pi_prime[:-1])
        self.pi *= pi_prime

        self.pi /= self.pi.sum()

    def _sample_gamma(self, eta_0):
        """
        """
        self.gamma = np.ones((self.num_glob_groups, self.max_num_layer_groups))
        for k in range(self.num_glob_groups):
            gamma_prime = np.random.beta(a=1, b=eta_0, size=(self.max_num_layer_groups, ))
            self.gamma[k,1:] = np.cumprod(1 - gamma_prime[:-1])
            self.gamma[k,:] *= gamma_prime

        row_sums = self.gamma.sum(axis=1, keepdims=True)
        self.gamma /= row_sums
        
    def _sample_sigma2(self, nu_0, omega_0):
        """
        """
        self.sigma2 = invgamma.rvs(nu_0, omega_0, size=(self.num_glob_groups, ))

    def _sample_phi0(self, mu):
        """
        """
        # num_feats = self.features.shape[1]
        # self.mu = np.random.multivariate_normal(mean=np.zeros(num_feats),
        #                                    cov=np.diag(np.ones(num_feats)))
        self.phi0 = np.random.multivariate_normal(mean=mu,
                                                  cov=np.eye(self.P),
                                                  size=self.num_glob_groups)

    def _sample_phi(self):
        """
        """
        # self.phi = np.random.multivariate_normal(
        #     mean=self.mu,
        #     cov=np.diag(np.ones(len(self.mu))),
        #     size=self.num_glob_groups
        # )
        self.phi = np.zeros((self.num_glob_groups, self.P))
        for k in range(self.num_glob_groups):
            self.phi[k,:] = np.random.multivariate_normal(
                mean=self.phi0[k,:],
                cov=self.sigma2[k] * np.eye(self.P))

    def _compute_delta(self):
        """
        """
        self.delta = np.zeros((self.num_nodes, self.num_glob_groups))
        for i in range(self.num_nodes):
            for m in range(self.num_glob_groups):
                self.delta[i,m] = np.dot(self.features[i,:], self.phi[m,:])

    def _compute_tau(self):
        """
        """
        self.tau = np.ones((self.num_nodes, self.num_glob_groups))
        for i in range(self.num_nodes):
            self.tau[i,1:] = np.cumprod(1 - norm.cdf(self.delta[i,:-1]))
            self.tau[i,:] *= norm.cdf(self.delta[i,:])

        row_sums = self.tau.sum(axis=1, keepdims=True)  
        self.tau /= row_sums

    def _sample_zeta(self):
        """
        """
        self.zeta = np.zeros((self.num_glob_groups, self.max_num_layer_groups))
        for k in range(self.num_glob_groups):
            for r in range(self.max_num_layer_groups):
                self.zeta[k,r] = np.random.choice(
                    np.arange(self.max_num_layer_groups),
                    p=self.pi
                )

    def _sample_global_groups(self):
        """
        """
        self.glob_groups = np.zeros((self.num_nodes, ), dtype=int)
        for i in range(self.num_nodes):
            self.glob_groups[i] = np.random.choice(
                np.arange(self.num_glob_groups), 
                p=self.tau[i,:])

    def _sample_u(self):
        """
        """
        self.u = np.zeros((self.num_layers, self.num_nodes), dtype=int)
        for l in range(self.num_layers):
            for i in range(self.num_nodes):
                self.u[l,i] = (
                    np.random.choice(
                        np.arange(self.max_num_layer_groups),
                        p=self.gamma[self.glob_groups[i],:])
                )

    def _compute_layer_groups(self):
        """
        """
        self.layer_groups = np.zeros((self.num_layers, self.num_nodes),
                                     dtype=int)
        for l in range(self.num_layers):
            for i in range(self.num_nodes):
                self.layer_groups[l,i] = self.zeta[self.glob_groups[i],
                                                   self.u[l,i]]
    
    def _sample_rho(self, alpha_0, beta_0):
        """
        """
        M = max(self.num_glob_groups, self.max_num_layer_groups)
        self.rho_matrix = np.random.beta(a=alpha_0, b=beta_0,
                                  size=(M, M))

    def _sample_adjacency_tensor(self):
        """
        """
        self.adjacency_tensor = np.zeros(
            (self.num_layers, self.num_nodes,  self.num_nodes), dtype=int)

        layer_rho_matrices = [
            self.rho_matrix[self.layer_groups[l, :].reshape(-1, 1), self.layer_groups[l, :]]
            for l in range(self.num_layers)
        ]

        for l in range(self.num_layers):
            for rep in range(self.num_adj_samps):
                random_nums = np.random.uniform(size=(self.num_nodes, self.num_nodes))
                adjacency_tensor_temp = (
                    random_nums < layer_rho_matrices[l]
                ).astype(int)
                self.adjacency_tensor[l,:,:] += adjacency_tensor_temp

    def sample_network(self, xi_0=None, eta_0=None, alpha_0=None, beta_0=None, mu=None):
        """
        """
        if self.specify:
            self._sample_adjacency_tensor()
        else:
            self._sample_pi(xi_0=xi_0)
            if self.gamma is None:
                self._sample_gamma(eta_0=eta_0)
            if self.phi is None:
                self._sample_phi0(mu=mu)
                self._sample_phi()
            self._compute_delta()
            self._compute_tau()
            self._sample_zeta()
            self._sample_global_groups()
            self._sample_u()
            self._compute_layer_groups()
            if self.rho_matrix is None:
                self._sample_rho(alpha_0=alpha_0, beta_0=beta_0)
            self._sample_adjacency_tensor()
        

