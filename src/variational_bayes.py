import numpy as np

from scipy.special import digamma
from joblib import Parallel, delayed
from scipy.stats import norm

class VariationalBayes:

    def __init__(self, num_nodes, num_layers, adj_tensor, features,
                 M_w, M_u, M_zeta) -> None:
        """
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.adj_tensor = adj_tensor
        self.features = features
        self.P = features.shape[1]
        self.M_w = M_w; self.M_u = M_u; self.M_zeta = M_zeta
    
        # Initialise empty arrays for the variational parameters.
        self.phi_u = np.zeros((num_layers, num_nodes, M_u))
        self.phi_w = np.zeros((num_nodes, M_w))
        self.phi_zeta = np.zeros((M_w, M_u, M_zeta))
        self.theta_delta = np.zeros((num_nodes, M_w))
        self.sigma2_delta = np.zeros((num_nodes, M_w)) # This is sigma^2 not sigma
        self.theta_phi = np.zeros((self.P, M_w))
        self.sigma_phi = np.zeros((self.P, M_w, M_w))
        self.theta_mu = None
        self.sigma_mu = None
        self.nu_sigma2 = None
        self.omega_sigma2 = None
        self.mu = None
        self.nu_0 = None
        self.omega_0 = None
        self.alpha_gamma = np.zeros((M_w, M_u))
        self.beta_gamma = np.zeros((M_w, M_u))
        self.alpha_pi = np.zeros((M_zeta, ))
        self.beta_pi = np.zeros((M_zeta, ))
        self.alpha_rho = np.zeros((M_zeta, M_zeta))
        self.beta_rho = np.zeros((M_zeta, M_zeta))

    def _update_q_u(self):
        """
        """
        # Precompute frequently used terms outside the loop
        precomputed_digamma_alpha_rho = (
            digamma(self.alpha_rho) - digamma(self.alpha_rho + self.beta_rho)
        )
        precomputed_digamma_beta_rho = (
            digamma(self.beta_rho) - digamma(self.alpha_rho + self.beta_rho)
        )

        # Non-parallelised computations
        term = np.einsum('iw,wu->iu', self.phi_w,
                         digamma(self.alpha_gamma) - 
                         digamma(self.alpha_gamma + 
                                 self.beta_gamma))
        cumsum = (
            np.cumsum(digamma(self.beta_gamma) -
                      digamma(self.alpha_gamma + 
                              self.beta_gamma), axis=1) -
            (digamma(self.beta_gamma) -
             digamma(self.alpha_gamma +
                     self.beta_gamma))
        ) 
        term += np.einsum('iw,wu->iu', self.phi_w, cumsum)
        
        term = np.tile(term, (self.num_layers, 1, 1))
        
        # Function for paraellelising the computations
        def compute_einsum_terms(l, i):
            local_term = np.zeros(self.M_u)

            ## For comparison with paper: w'=x, u'=v, r'=s
            # adj_tensor_mask defined to ensure not summing over j=i
            adj_tensor_mask = self.adj_tensor[l,i,:].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('w,jx,jv,wur,xvs,j,rs->u',
                                    self.phi_w[i, :], self.phi_w, 
                                    self.phi_u[l, :, :],
                                    self.phi_zeta, self.phi_zeta,
                                    adj_tensor_mask,
                                    precomputed_digamma_alpha_rho,
                                    optimize=True)

            local_term += np.einsum('w,jx,jv,wur,xvs,j,sr->u',
                                    self.phi_w[i, :], self.phi_w, 
                                    self.phi_u[l, :, :],
                                    self.phi_zeta, self.phi_zeta,
                                    adj_tensor_mask,
                                    precomputed_digamma_alpha_rho,
                                    optimize=True)

            adj_tensor_mask = 1 - self.adj_tensor[l,i,:].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('w,jx,jv,wur,xvs,j,rs->u',
                                    self.phi_w[i, :], self.phi_w, 
                                    self.phi_u[l, :, :],
                                    self.phi_zeta, self.phi_zeta,
                                    adj_tensor_mask,
                                    precomputed_digamma_beta_rho,
                                    optimize=True)

            local_term += np.einsum('w,jx,jv,wur,xvs,j,sr->u',
                                    self.phi_w[i, :], self.phi_w, 
                                    self.phi_u[l, :, :],
                                    self.phi_zeta, self.phi_zeta,
                                    adj_tensor_mask,
                                    precomputed_digamma_beta_rho,
                                    optimize=True)

            return local_term
        
        # Parallel computation 
        term_updates = Parallel(n_jobs=-1)(delayed(compute_einsum_terms)(l, i) 
                                            for l in range(self.num_layers) 
                                            for i in range(self.num_nodes))

        # Add results back to phi_u
        term += np.array(term_updates).reshape(self.num_layers, 
                                                    self.num_nodes, 
                                                    -1)
        self.phi_u = term

    def _delta_sampler(self, num_mc):
        """
        Function to sample values for delta for the current estimates of
        the variationl parameters.
        """
        delta_ik_samps = np.zeros((self.num_nodes, self.M_w, num_mc))
        rng = np.random.default_rng() # Speed improvement with this sampler
        for i in range(self.num_nodes):
            for k in range(self.M_w):
                delta_ik_samps[i,k,:] = rng.normal(self.theta_delta[i,k],
                                                   np.sqrt(self.sigma_delta[i,k]),
                                                   size=num_mc)
                
        # Optional with JIT to speed-up computation
        # delta = np.random.zeros((self.num_nodes, M_delta, num_mc))
        # @njit(parallel=True)
        # def generate_delta(delta, a, b):
        #     for i in prange(1000):  # prange enables parallel loops
        #         for k in range(10):
        #             delta[i, k, :] = np.random.normal(a[i, k], b[i, k], size=(num_mc))
        #     return delta
        
        # self.delta_ik_samps = generate_delta(self.delta_ik_samps, 
        #                                      self.theta_delta, 
        #                                      np.sqrt(self.sigma_delta))

        return delta_ik_samps
    
    def _update_q_w(self, num_mc):
        """
        """
        cumsum = (
            np.cumsum(digamma(self.beta_gamma) -
                      digamma(self.alpha_gamma + 
                              self.beta_gamma), axis=1) -
            (digamma(self.beta_gamma) -
             digamma(self.alpha_gamma +
                     self.beta_gamma))
        )
        cumsum += (digamma(self.beta_gamma) -
                   digamma(self.alpha_gamma +
                           self.beta_gamma)
        )

        term = np.einsum('liu,wu->iw', self.phi_u, cumsum)

        # MC step for the expectation
        delta_ik_samps = self._delta_sampler(num_mc)  

        # Compute the approximation of the expectation using the samples
        tau = np.zeros((self.num_nodes, self.M_w, num_mc))
        tau[:,1:,:] = (1 - norm.cdf(delta_ik_samps[:,:-1,:]).cumprod(axis=1))
        tau *= norm.cdf(delta_ik_samps)

        term += np.log(tau).mean(axis=2) 

        def compute_einsum_terms(i):
            adj_tensor_mask = self.adj_tensor[:, i, :].copy()
            adj_tensor_mask[:, i] = 0
            adj_tensor_sub_mask = 1 - self.adj_tensor[:, i, :].copy()
            adj_tensor_sub_mask[:, i] = 0

            # w'=x, u'=v, r'=s
            local_term = np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
                                   self.phi_w, self.phi_u, self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_mask, 
                                   digamma(self.alpha_rho) - 
                                   digamma(self.alpha_rho + self.beta_rho))
            local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
                                   self.phi_w, self.phi_u, self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_sub_mask, 
                                   digamma(self.beta_rho) - 
                                   digamma(self.alpha_rho + self.beta_rho))
            
            adj_tensor_mask = self.adj_tensor[:, :, i].copy()
            adj_tensor_mask[:, i] = 0
            adj_tensor_sub_mask = 1 - self.adj_tensor[:, :, i].copy()
            adj_tensor_sub_mask[:, i] = 0
            local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
                                   self.phi_w, self.phi_u, self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_mask, 
                                   digamma(self.alpha_rho.T) - 
                                   digamma(self.alpha_rho.T + self.beta_rho.T))
            local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
                                   self.phi_w, self.phi_u, self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_sub_mask, 
                                   digamma(self.beta_rho.T) - 
                                   digamma(self.alpha_rho.T + self.beta_rho.T))
            
            return local_term   

        parallel_terms = Parallel(n_jobs=-1)(delayed(compute_einsum_terms)(i) 
                        for i in range(self.num_nodes)) 
        term += np.array(parallel_terms).reshape(self.num_nodes, self.M_w, -1)

    def _update_q_zeta(self):
        """
        """
        term = (
            digamma(self.alpha_pi) - digamma(self.beta_pi) +
            np.cumsum(digamma(self.beta_pi) - 
                      digamma(self.alpha_pi + self.beta_pi))
        )
        term = np.tile(term, (self.M_w, self.M_u, self.M_zeta)) # !! M_zeta3 = M_pi !!

        def compute_einsum_terms(k, r):
            """
            """
            # Masks for ensuring we don't sum over unintended indices
            mask_k = np.ones((self.M_w,)) 
            mask_k[k] = 0
            mask_r = np.ones((self.M_u,)) 
            mask_r[r] = 0
            adj_tensor_mask = self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_mask[l], 0)
            adj_tensor_sub_mask = 1 - self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_sub_mask[l], 0)

            term = np.zeros(self.M_zeta) 

            term += np.einsum('i,j,li,lj,lij->', self.phi_w[:,k],
                                    self.phi_w[:,k], self.phi_u[:,:,r],
                                    self.phi_u[:,:,r], adj_tensor_mask, 
                                    optimize=True)
            term *= np.diag(
                digamma(self.alpha_rho) - 
                digamma(self.alpha_rho + self.beta_rho)
            )

            term += np.einsum('i,j,li,lj,lij->', self.phi_w[:,k],
                                    self.phi_w[:,k], self.phi_u[:,:,r],
                                    self.phi_u[:,:,r], adj_tensor_sub_mask, 
                                    optimize=True)
            term *= np.diag(
                digamma(self.beta_rho) - 
                digamma(self.alpha_rho + self.beta_rho)
            )

            # s' = t
            term += np.einsum('i,j,li,lj,i,w,jw,li,u,lju,wut,lij,st->s', 
                            self.phi_w[:,k], self.phi_w[:,k],
                            self.phi_u[:,:,r], self.phi_u[:,:,r],
                            self.phi_w[:,k], mask_k, self.phi_w,
                            self.phi_u[:,:,r],mask_r, self.phi_u,
                            self.phi_zeta, adj_tensor_mask,
                            digamma(self.alpha_rho) - 
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)    

            term += np.einsum('i,j,li,lj,i,w,jw,li,u,lju,wut,lij,st->s', 
                            self.phi_w[:,k], self.phi_w[:,k],
                            self.phi_u[:,:,r], self.phi_u[:,:,r],
                            self.phi_w[:,k], mask_k, self.phi_w,
                            self.phi_u[:,:,r],mask_r, self.phi_u,
                            self.phi_zeta, adj_tensor_sub_mask,
                            digamma(self.beta_rho) - 
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)
            
            term += np.einsum('i,j,li,lj,i,w,jw,li,u,lju,wut,lji,ts->s', 
                            self.phi_w[:,k], self.phi_w[:,k],
                            self.phi_u[:,:,r], self.phi_u[:,:,r],
                            self.phi_w[:,k], mask_k, self.phi_w,
                            self.phi_u[:,:,r],mask_r, self.phi_u,
                            self.phi_zeta, adj_tensor_mask,
                            digamma(self.alpha_rho) - 
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)
            
            term += np.einsum('i,j,li,lj,i,w,jw,li,u,lju,wut,lji,ts->s', 
                            self.phi_w[:,k], self.phi_w[:,k],
                            self.phi_u[:,:,r], self.phi_u[:,:,r],
                            self.phi_w[:,k], mask_k, self.phi_w,
                            self.phi_u[:,:,r],mask_r, self.phi_u,
                            self.phi_zeta, adj_tensor_sub_mask,
                            digamma(self.beta_rho) -
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)

            return term

        parallel_terms = Parallel(n_jobs=-1)(delayed(compute_einsum_terms)(k, r) 
                        for k in range(self.M_w) 
                        for r in range(self.M_u)) 
        term += np.array(parallel_terms).reshape(self.M_w, self.M_u, -1)

        self.phi_zeta = term

    def _update_q_delta(self, num_mc, num_grad_steps, 
                        alpha=0.001, beta1=0.9, beta2=0.999, eps=10**-8):
        """
        Uses an ADAM optimiser to go
        """
        # Dot products
        X_dot_X = np.sum(self.features ** 2, axis=1) # Shape (self.num_nodes, )
        X_dot_theta_phi = self.features @ self.theta_phi.T # Shape (self.num_nodes, M_phi)

        # ADAM parameters
        first_moment_theta = np.zeros((self.num_nodes, self.M_w))
        first_moment_sigma2 = np.zeros((self.num_nodes, self.M_w))
        second_moment_theta = np.zeros((self.num_nodes, self.M_w))
        second_moment_sigma2 = np.zeros((self.num_nodes, self.M_w))

        for step in range(num_grad_steps):
            # Skip the zero-index
            step += 1

            # Empty arrays for gradients
            grad_theta = np.zeros((self.num_nodes, self.M_w))
            grad_sigma2 = np.zeros((self.num_nodes, self.M_w))

            # Sample delta using current variational parameter values
            delta_ik_samps = self._delta_sampler(num_mc)

            # Approximate the expectation using MC
            expectation_theta_mc = (
                (delta_ik_samps - np.tile(self.theta_delta[:,:,np.newaxis], (1,1,num_mc)))
                * np.log(norm.cdf(delta_ik_samps)).mean(axis=2)
            ) # Shape (self.num_nodes, M_delta)

            expectation_sigma_mc = (
                ((delta_ik_samps - np.tile(self.theta_delta[:,:,np.newaxis], (1,1,num_mc))) ** 2 
                - np.tile(self.sigma2_delta[:,:,np.newaxis], (1,1,num_mc)) ** 2) * 
                np.log(norm.cdf(delta_ik_samps)).mean(axis=2)
            ) # Shape (self.num_nodes, M_delta)

            grad_theta = (
                -(self.nu_sigma2 / self.omega_sigma2) * (1 / X_dot_X[:,np.newaxis]) * self.theta_delta 
                + self.nu_sigma2 / self.omega_sigma2 * X_dot_theta_phi / (1 / X_dot_X[:,np.newaxis])
                + (1 / self.sigma2_delta) * expectation_theta_mc *
                    np.cumsum(self.phi_w[:,::-1], axis=1)[:, ::-1]
            )

            grad_sigma2 = (
                -(self.nu_sigma2 / self.omega_sigma2) * (1 / (2 * X_dot_X[:,np.newaxis]))
                + (1 / (2 * self.sigma2_delta ** 2)) * expectation_sigma_mc * 
                    np.cumsum(self.phi_w[:,::-1], axis=1)[:, ::-1]
            )

            # ADAM steps
            first_moment_theta = beta1 * first_moment_theta + (1 - beta1) * grad_theta
            second_moment_theta = beta2 * second_moment_theta + (1 - beta2) * grad_theta ** 2
            first_moment_sigma2 = beta1 * first_moment_sigma2 + (1 - beta1) * grad_sigma2
            second_moment_sigma2 = beta2 * second_moment_sigma2 + (1 - beta2) * grad_sigma2 ** 2

            first_moment_theta_bias = first_moment_theta / (1 - beta1 ** step)
            second_moment_theta_bias = second_moment_theta / (1 - beta2 ** step)
            first_moment_sigma2_bias = first_moment_sigma2 / (1 - beta1 ** step)
            second_moment_sigma2_bias = second_moment_sigma2 / (1 - beta2 ** step)
    
            self.theta_delta = (
                self.theta_delta 
                + alpha * first_moment_theta_bias / (np.sqrt(second_moment_theta_bias) + eps)
            )
            self.sigma2_delta = (
                self.sigma2_delta 
                + alpha * first_moment_sigma2_bias / (np.sqrt(second_moment_sigma2_bias) + eps)
            ) # Note the + here as we try to maximise the ELBO

    def _update_q_phi(self):
        """
        """
        # Compute \sum_i x_ix_i^T/(x_i^Tx_i)
        outer_products = np.einsum('ij,ik->ijk', self.features, self.features)
        normalisation_terms = np.einsum('ij,ij->i', self.features, self.features)
        # For Sigma computation
        normalised_matrix = (
            np.sum(outer_products / normalisation_terms[:, np.newaxis, np.newaxis], axis=0)
        )
        # For theta computation
        pre_multiplied = (
            self.mu.reshape(-1,1) + np.einsum('k,ik,ip,i->kp', self.nu_sigma2 / self.omega_sigma2, 
                                              self.theta_delta, self.features, 1/normalisation_terms)
        )

        for k in range(self.P):
            self.sigma_phi[k,:,:] = np.linalg.inv(
                np.eye(self.P) + self.nu_sigma2[k] / self.omega_sigma2[k] * normalised_matrix
            )
            self.theta_phi[k,:] = self.sigma_phi[k,:,:] @ pre_multiplied[k,:]

    def _update_q_gamma(self):
        """
        """
        def compute_einsum_terms_alpha(k, s):
            return 1 + np.einsum('li,i->',
                                 self.phi_u[:,:,s],
                                 self.phi_w[:,k])
        
        def compute_einsum_terms_beta(k, s):
            # Mask for sum from r=s+1 to M_u only
            mask_s = np.zeros((self.M_u,))
            mask_s[(s+1):] = 1

            return self.eta_0 + np.einsum('r,lir,i->',
                                 mask_s,
                                 self.phi_u,
                                 self.phi_w[:,k])
        
        term = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_alpha)(k, s) 
                   for k in range(self.M_w) 
                   for s in range(self.M_u)) 
        
        self.alpha_gamma = term.reshape((self.M_w, self.M_u))

        term = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_beta)(k, s) 
                   for k in range(self.M_w) 
                   for s in range(self.M_u)) 

        self.beta_gamma = term.reshape((self.M_w, self.M_u))

    def _update_q_pi(self):
        """
        """
        self.alpha_pi = 1 + np.einsum('krs->s', self.phi_zeta)
        beta_temp = np.einsum('krm->m', self.phi_zeta)
        self.beta_pi = self.xi_0 + beta_temp[::-1].cumsum()[::-1] - beta_temp

    def _update_q_rho(self):
        """
        """
        def compute_einsum_terms_alpha(l,i):
            adj_tensor_mask = self.adj_tensor[l,i,:].copy()
            adj_tensor_mask[i] = 0   

            # w' = x, u' = v
            local_term = np.einsum('j,wuk,xvm,w,jx,u,jv->km',
                                   adj_tensor_mask, self.phi_zeta,
                                   self.phi_zeta, self.phi_w[i, :],
                                   self.phi_w, self.phi_u[l, i, :],
                                   self.phi_u[l, :, :]
            )                                 

            return local_term

        def compute_einsum_terms_beta(l,i):
            adj_tensor_sub_mask = 1 - self.adj_tensor[l,i,:].copy()
            adj_tensor_sub_mask[i] = 0  

            # w' = x, u' = v
            local_term = np.einsum('j,wuk,xvm,w,jx,u,jv->km',
                                   adj_tensor_sub_mask, self.phi_zeta,
                                   self.phi_zeta, self.phi_w[i, :],
                                   self.phi_w, self.phi_u[l, i, :],
                                   self.phi_u[l, :, :]
            )    

            return local_term
        
        # Parallel computation 
        term_updates = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_alpha)(l, i) 
                                            for l in range(self.num_layers) 
                                            for i in range(self.num_nodes))

        # Add results back to alpha_rho
        self.alpha_rho = self.alpha_0
        self.alpha_rho += np.array(term_updates).reshape(self.M_zeta, 
                                                         self.M_zeta, 
                                                         -1).sum(axis=2)
        
        term_updates = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_beta)(l, i) 
                                            for l in range(self.num_layers) 
                                            for i in range(self.num_nodes))

        # Add results back to beta_rho
        self.beta_rho = self.beta_0
        self.beta_rho += np.array(term_updates).reshape(self.M_zeta, 
                                                        self.M_zeta, 
                                                        -1).sum(axis=2)
    
    def _compute_q_mu(self):
        """
        """
        self.theta_mu = (self.theta_phi + self.mu) / 2
        self.sigma_mu = 2 * np.eye(self.M_w)

    def _compute_q_sigma2(self):
        """
        """
        self.nu_sigma2 = self.nu_0 + self.num_nodes / 2
        self.omega_sigma2 = (
            self.omega_0 + np.sum(
                ((self.theta_delta - self.features @ self.theta_phi.T) ** 2 + self.sigma2_delta)
                / (2 * np.sum(self.features ** 2, axis=1)), axis=0)
        )

    def run_VB_scheme(self, num_mc: int):
        """
        Run the full VB update scheme.
        Parameter:
            - num_mc: number of MC samples for expectation estimates.
        """
