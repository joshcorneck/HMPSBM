import numpy as np

from scipy.special import gamma, loggamma, digamma, logsumexp
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import src.helper_functions.initialise_parameters as initialise
from src.helper_functions.compute_ELBO import *

class VariationalBayes:

    def __init__(self, num_nodes: int, num_layers: int, adj_tensor: np.array, 
                 features: np.array, M_w: int, M_u: int, 
                 num_fp_its: int=1, degree_correction: bool=False, k_max: int = 4) -> None:
        """
        A class to compute a mean-field variational approximation to the posterior.
        Parameters:
            - num_nodes: number of nodes in the network (N).
            - num_layers: number of layers in the network (L).
            - adj_tensor: adjacency tensor for the network, of shape (L,N,N).
            - features: features for each node of the network, of shape (N,P).
            - M_w, M_u: values for variational approximation truncation.
            - k_max: embedding dimension used for initialisation.
        """
        self.num_fp_its = num_fp_its
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.adj_tensor = adj_tensor
        self.features = features
        self.P = features.shape[1]
        self.M_w = M_w; self.M_u = M_u        
        self.k_max = k_max
            
        self.nu_sigma2 = np.ones((M_w, ))
        self.omega_sigma2 = np.ones((M_w, ))
        self.mu = np.zeros((self.P, ))
        
        self.nu_0 = 1
        self.omega_0 = 1
        self.alpha_0 = 1
        self.beta_0 = 1
        self.eta_0 = 1 / self.M_u
        
    def _initialise_parameters(self):
        """
        Initialise all parameter values.
        """
        # Initialise phi_u
        self.phi_u = initialise.initialise_phi_u(self.num_nodes, self.num_layers,
                                                 self.k_max, self.M_u, self.adj_tensor)
        
        # Initialise phi_w
        self.phi_w = initialise.initialise_phi_w(self.num_nodes, self.num_layers, 
                                                 self.k_max, self.M_w, self.features, 
                                                 self.adj_tensor)
        
        # Initialise alpha_rho and beta_rho
        self.alpha_rho, self.beta_rho = (
            initialise.initialise_alpha_beta_rho(self.num_layers, self.M_u, self.adj_tensor,
                                                 self.phi_u)
        )
        
        # Initialise alpha_gamma and beta_gamma
        self.alpha_gamma, self.beta_gamma = (
            initialise.initialise_alpha_beta_gamma(self.adj_tensor, self.phi_w,
                                                   self.phi_u)
        )
        
        # Initialise theta_phi, theta_phi0, sigma_phi and sigma_phi0
        self.theta_phi = np.stack(initialise.initialise_theta_phi(self.features, self.phi_w))
        self.theta_phi0 = np.ones((self.M_w, self.P))
        
        cov_mat = np.cov(self.features.T)
        self.sigma_phi = np.ones((self.M_w, self.P, self.P))
        self.sigma_phi0 = np.ones((self.M_w, self.P, self.P))
        for m in range(self.M_w):
            self.sigma_phi0[m,:,:] = cov_mat
            self.sigma_phi[m,:,:] = cov_mat
            
    
    def _update_q_u(self):
        """
        A method for computing the variational approximation for each u_{\ell i}
        """
        # Precompute frequently used terms outside the loop (avoiding zeros)
        precomputed_digamma_alpha_rho = np.zeros_like(self.alpha_rho)
        precomputed_digamma_beta_rho = np.zeros_like(self.beta_rho)
        valid_mask = (self.alpha_rho  > 0) & (self.beta_rho > 0)
        precomputed_digamma_alpha_rho[valid_mask] = (
            digamma(self.alpha_rho[valid_mask]) 
            - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
        )
        precomputed_digamma_beta_rho[valid_mask] = (
            digamma(self.beta_rho[valid_mask]) 
            - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
        )

        # Non-parallelised computations
        cumsum = (
            digamma(self.alpha_gamma) - 
            digamma(self.alpha_gamma + 
                    self.beta_gamma) + 
            np.cumsum(digamma(self.beta_gamma) -
                      digamma(self.alpha_gamma + 
                              self.beta_gamma), axis=1) -
            (digamma(self.beta_gamma) -
             digamma(self.alpha_gamma +
                     self.beta_gamma))
        )
        
        term = np.einsum('iw,wu->iu', self.phi_w, cumsum)
        term = np.tile(term, (self.num_layers, 1, 1))
        
        # Function for paraellelising the computations
        def compute_einsum_terms_u(l, i, term):
            # THIS IS TO BE REFORMATTED.
            local_term = term.copy()

            ## For comparison with paper: w'=x, u'=v, r'=s
            # adj_tensor_mask defined to ensure not summing over j=i
            adj_tensor_mask = self.adj_tensor[l, i, :].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('jm,j,km->k', 
                                    self.phi_u[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_alpha_rho,
                                    optimize=True)

            adj_tensor_mask = self.adj_tensor[l, :, i].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('jm,j,mk->k',
                                    self.phi_u[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_alpha_rho,
                                    optimize=True)

            adj_tensor_mask = 1 - self.adj_tensor[l,i,:].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('jm,j,km->k',
                                    self.phi_u[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_beta_rho,
                                    optimize=True)

            adj_tensor_mask = 1 - self.adj_tensor[l,:,i].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('jm,j,mk->k', 
                                    self.phi_u[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_beta_rho,
                                    optimize=True)

            return local_term
        
        for CAVI_rep in range(self.num_fp_its):
            for l in range(self.num_layers):
                for i in range(self.num_nodes):
                    phi_temp = compute_einsum_terms_u(l, i, term[l,i,:])
                    self.phi_u[l,i,:] = np.exp(phi_temp - logsumexp(phi_temp))
    
    def _compute_ELBO_wrt_u(self, CAVI_rep: int, idx: int):
        """
        """
        
        def compute_expected_log_joint():
            """
            """
            ELBO_temp = 0
            
            digamma_alpha_rho = np.zeros_like(self.alpha_rho)
            digamma_beta_rho = np.zeros_like(self.beta_rho)
            valid_mask = (self.alpha_rho  > 0) & (self.beta_rho > 0)
            digamma_alpha_rho[valid_mask] = (
                digamma(self.alpha_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            digamma_beta_rho[valid_mask] = (
                digamma(self.beta_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            
            digamma_alpha_gamma = (
                digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma)
                )
            digamma_beta_gamma = (
                digamma(self.beta_gamma) - digamma(self.beta_gamma + self.beta_gamma)
                )
            
            # Term 1: p(A | z, \rho)
            adj_tensor_mask = self.adj_tensor.copy()
            adj_tensor_sub_mask = self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_mask[l,:,:], 0)
                np.fill_diagonal(adj_tensor_sub_mask[l,:,:], 0)
            
            ELBO_temp += np.einsum('lik,ljm,lij,km->', self.phi_u, self.phi_u, 
                              adj_tensor_mask, digamma_alpha_rho,
                              optimize=True)
            
            ELBO_temp += np.einsum('lik,ljm,lij,km->', self.phi_u, self.phi_u, 
                              adj_tensor_sub_mask, digamma_beta_rho,
                              optimize=True)
            
            # Term 3: p(z | w, \gamma)
            cumsum_gamma = (
                digamma_alpha_gamma + 
                np.cumsum(digamma_beta_gamma, axis=1) -
                digamma_beta_gamma
                )
                
            ELBO_temp += np.einsum('lik,iw,wk->', self.phi_u, self.phi_w,
                              cumsum_gamma,
                              optimize=True)
            
            return ELBO_temp
        
        def compute_expected_log_q():
            """
            """
            ELBO_temp = 0
            
            # Term 1: q(z)
            ELBO_temp += (self.phi_u * np.log(self.phi_u + 10e-10)).sum()
            
            return ELBO_temp
        
        # Compute and store
        self.ELBO_store_u[CAVI_rep, idx] = (
            compute_expected_log_joint() - compute_expected_log_q()
        )
        
    def _sample_phi(self, num_mc: int):
        """
        A method to sample values for phi for the current estimates of
        the variationl parameters.
        Parameters:
            - num_mc: the number of MC samples.
        """
        phi_k_samps = np.zeros((self.M_w, self.P, num_mc))
        rng = np.random.default_rng() # Speed improvement with this sampler
        for k in range(self.M_w):
            phi_k_samps[k,:,:] = rng.multivariate_normal(self.theta_phi[k,:],
                                                         self.sigma_phi[k,:,:],
                                                         size=num_mc).T
                
        return phi_k_samps
    
    def _compute_delta(self, num_mc: int, phi_k_samps: np.array = None):
        """
        """
        if phi_k_samps is None:
            phi_k_samps = self._sample_phi(num_mc)
        
        delta_ik_samps = np.zeros((self.num_nodes, self.M_w, num_mc))
        for i in range(self.num_nodes):
            for k in range(self.M_w):
                delta_ik_samps[i,k,:] = np.dot(self.features[i,:], phi_k_samps[k])
                
        return delta_ik_samps
    
    def _compute_tau(self, num_mc: int, phi_k_samps: np.array = None):
        """
        """
        if phi_k_samps is None:
            phi_k_samps = self._sample_phi(num_mc)
        
        delta_ik_samps = np.zeros((self.num_nodes, self.M_w, num_mc))
        for i in range(self.num_nodes):
            for k in range(self.M_w):
                delta_ik_samps[i,k,:] = np.dot(self.features[i,:], phi_k_samps[k])
                
        tau_ik_samps = np.zeros_like(delta_ik_samps)
        for i in range(self.num_nodes):
            for k in range(self.M_w):
                if k == 0:
                    tau_ik_samps[i,k,:] = norm.cdf(delta_ik_samps[i,k,:])
                else:
                    tau_ik_samps[i,k,:] = (
                        norm.cdf(delta_ik_samps[i,k,:]) *
                        (1 - norm.cdf(delta_ik_samps[i,:k,:])).prod(axis=0)
                    )
                    
        # Normalise
        row_sums = tau_ik_samps.sum(axis=1, keepdims=True)

        # Avoid division by zero by setting zero sums to 1 temporarily
        row_sums[row_sums == 0] = 1

        # Perform element-wise division, rows with original zero sums will remain unchanged
        tau_ik_samps = tau_ik_samps / row_sums
                
        return tau_ik_samps
    
    def _update_q_w(self, num_mc: int):
        """
        A method for computing the variational approximation for each w_i.
        Parameters:
            - num_mc: the number of MC samples used for the MC approximation to the 
                      expectation of delta.
        """
        cumsum = (
            digamma(self.alpha_gamma) - 
            digamma(self.alpha_gamma + 
                    self.beta_gamma) +
            np.cumsum(digamma(self.beta_gamma) -
                      digamma(self.alpha_gamma + 
                              self.beta_gamma), axis=1) -
            (digamma(self.beta_gamma) -
             digamma(self.alpha_gamma +
                     self.beta_gamma))
        )
    
        term = np.einsum('liu,wu->iw', self.phi_u, cumsum)

        # MC step for the expectation
        tau_ik_samps = self._compute_tau(num_mc)
        tau_expec = np.log(tau_ik_samps + 10e-10).mean(axis=2)
        term += tau_expec
    
        # Take exponential and normalise
        for i in range(self.num_nodes):
            self.phi_w[i,:] = np.exp(term[i,:] - logsumexp(term[i,:]))
            
    def _compute_ELBO_wrt_w(self, CAVI_rep: int, num_mc: int, idx: int):
        """
        """
        def compute_expected_log_joint():
            """
            """
            ELBO_temp = 0

            digamma_alpha_gamma = (
                digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma)
                )
            digamma_beta_gamma = (
                digamma(self.beta_gamma) - digamma(self.beta_gamma + self.beta_gamma)
                )
            
            # Term 3: p(z | w, \gamma)
            cumsum_gamma = (
                digamma_alpha_gamma + 
                np.cumsum(digamma_beta_gamma, axis=1) -
                digamma_beta_gamma
                )
                
            ELBO_temp += np.einsum('lik,iw,wk->', self.phi_u, self.phi_w,
                              cumsum_gamma,
                              optimize=True)
            
            # Term 4: p(w | \tau)
            # MC step for the expectation
            tau_ik_samps = self._compute_tau(num_mc)
            log_tau_expec = np.log(tau_ik_samps + 10e-10).mean(axis=2)
            
            ELBO_temp += np.einsum('iw,iw->', self.phi_w, log_tau_expec)
            
            return ELBO_temp
        
        def compute_expected_log_q():
            """
            """
            ELBO_temp = 0
    
            # Term 3: q(w)
            ELBO_temp += (self.phi_w * np.log(self.phi_w + 10e-10)).sum()
            
            return ELBO_temp
        
        # Compute and store
        self.ELBO_store_w[CAVI_rep, idx] = (
            compute_expected_log_joint() - compute_expected_log_q()
        )
        
    def _update_q_phi(self, num_mc: int, num_grad_steps: int, CAVI_rep: int,
                      alpha: float=0.001, beta1: float=0.9, beta2: float=0.999, 
                      eps: float=10**-8):
        """
        A method for computing the variational approximation for each \phi_{k}. This
        uses an ADAM optimiser to perform the gradient ascent steps to maximise the ELBO.
        Parameters:
            - num_mc: number of MC samples for expectation approximations.
            - num_grad_steps: number of ADAM steps to compute.
            - alpah, beta1, beta2, eps: standard ADAM parameters.
        """
        def _compute_M_from_sigma(sigma):
            """
            """
            L = np.linalg.cholesky(sigma)
            
            M = np.tril(L, k=-1) 
            np.fill_diagonal(M, np.log(np.diag(L)))
            
            return M
        
        def _compute_sigma_and_inv_from_M(M):
            """
            """
            L = np.tril(M, k=-1)
            np.fill_diagonal(L, np.exp(np.diag(M)))
            
            sigma = L @ L.T
            sigma_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
            
            return sigma, sigma_inv

        def _compute_L_from_M(M):
            """
            """
            L = np.tril(M, k=-1)
            np.fill_diagonal(L, np.exp(np.diag(M)))
            
            return L

        # ADAM parameters
        first_moment_theta = np.zeros((self.M_w, self.P))
        first_moment_M = np.zeros((self.M_w, self.P, self.P))
        second_moment_theta = np.zeros((self.M_w, self.P))
        second_moment_M = np.zeros((self.M_w, self.P, self.P))
        
        # Compute initial values of the M_m
        M = np.zeros((self.M_w, self.P, self.P))
        for m in range(self.M_w):
            M[m,:,:] = _compute_M_from_sigma(self.sigma_phi[m,:,:])

        for step in range(num_grad_steps):
            # Skip the zero-index
            step += 1
            
            print(f"Gradient step {step} of {num_grad_steps}.")

            # Empty arrays for gradients
            grad_theta = np.zeros((self.M_w, self.P))
            grad_sigma = np.zeros((self.M_w, self.P, self.P))
            grad_M = np.zeros((self.M_w, self.P, self.P))
            
            if step == 1:
                # Compute the sigma_phi and inverse of sigma_phi
                sigma_phi_inv = np.zeros((self.M_w, self.P, self.P))
                for m in range(self.M_w):
                    self.sigma_phi[m,:,:], sigma_phi_inv[m,:,:] = (
                        _compute_sigma_and_inv_from_M(M[m,:,:])
                    )
            
            # Sample phi and compute delta using current variational parameter values
            phi_k_samps = self._sample_phi(num_mc)
                            
            ## Estimate the expectation for theta
            # Compute the phi - theta
            vec = (phi_k_samps - np.tile(self.theta_phi[:, :, np.newaxis], (1, 1, num_mc))) # (M_w, P, num_mc)
            
            # Once with einsum
            einsum_term = np.einsum('np,mpc->nmc', self.features, phi_k_samps, optimize=True) # (num_nodes, M_w, num_mc)
            cdf_term = np.log(norm.cdf(einsum_term) + 10e-10) # Shift away from 0
            cdf_term_sub = np.log(1 - norm.cdf(einsum_term) + 10e-10) # Shift away from 0
                        
            # Multiply and avergage to get mc approx
            mc_approx = np.einsum('mpc,nmc->nmp', vec, cdf_term, optimize=True) / num_mc # (num_nodes, M_w, P)
            mc_approx_sub = np.einsum('mpc,nmc->nmp', vec, cdf_term_sub, optimize=True) / num_mc
                    
            # Scale by matrix
            scaled = np.einsum('mij,nmj->nmi', sigma_phi_inv, mc_approx, optimize=True) # (num_nodes, M_w, P)
            scaled_sub = np.einsum('mij,nmj->nmi', sigma_phi_inv, mc_approx_sub, optimize=True)
                        
            # Sum over phi_w
            grad_theta += np.einsum('nm,nmp->mp', self.phi_w, scaled)
            grad_theta += np.einsum('nm,nmp->mp',
                                    (np.cumsum(self.phi_w[:,::-1], axis=1)[:, ::-1] - self.phi_w), 
                                    scaled_sub)            
            # Add on remaining terms
            for m in range(self.M_w):
                grad_theta[m,:] += ((self.nu_sigma2[m] / self.omega_sigma2[m]) * 
                                    (self.theta_phi[m,:] - self.theta_phi0[m,:])
                )
                        
            # WE WANT TO MAXIMISE THE ELBO, SO APPLY ADAM TO THE NEGATIVE GRADIENT
            grad_theta = -grad_theta
                                    
            ## Estimate the expectation for sigma
            # Compute outer product and add matrices - need to rewrite in suffix notation
            outer_prod = np.zeros((self.M_w, self.P, self.P, num_mc))
            
            for m in range(self.M_w):
                for mc in range(num_mc):
                    outer_prod[m,:,:, mc] = np.outer(vec[m,:,mc], vec[m,:,mc])
            
            mc_approx = np.zeros((self.num_nodes, self.M_w, self.P, self.P))
            mc_approx_sub = np.zeros((self.num_nodes, self.M_w, self.P, self.P))
            
            for i in range(self.num_nodes):
                for m in range(self.M_w):
                    mc_approx[i, m, :, :] = np.sum(outer_prod[m, :, :, :] 
                                                   * cdf_term[i, m, :], axis=-1) / num_mc
                    mc_approx_sub[i, m, :, :] = np.sum(outer_prod[m, :, :, :] 
                                                       * cdf_term_sub[i, m, :], axis=-1) / num_mc

                    # Scale by sigma_phi_inv and 1/2
                    mc_approx[i, m, :, :] = 0.5 * (
                        sigma_phi_inv[m, :, :] @ mc_approx[i, m, :, :] @ sigma_phi_inv[m, :, :]
                        - np.mean(cdf_term[i, m, :]) * sigma_phi_inv[m, :, :]
                    )
                    mc_approx_sub[i, m, :, :] = 0.5 * (
                        sigma_phi_inv[m, :, :] @ mc_approx_sub[i, m, :, :] @ sigma_phi_inv[m, :, :]
                        - np.mean(cdf_term_sub[i, m, :]) * sigma_phi_inv[m, :, :]
                    )
                                
            # Sum over phi_w
            for m in range(self.M_w):
                for i in range(self.num_nodes):
                    grad_sigma[m,:,:] += self.phi_w[i,m] * mc_approx[i,m,:,:]
            
            for m in range(self.M_w):
                for i in range(self.num_nodes):
                    for k in range(m+1, self.M_w):
                        grad_sigma[m,:,:] += self.phi_w[i,k] * mc_approx_sub[i,m,:,:]
                        
            # Add on the remaining terms
            for m in range(self.M_w):
                grad_sigma += 0.5 * (
                    self.nu_sigma2[m] / self.omega_sigma2[m] * np.eye(self.P) +
                    sigma_phi_inv[m,:,:]
                )
            
            # Convert from sigma_m gradient to M_m gradient
            for m in range(self.M_w):
                grad_sigma_M_m = np.zeros(shape=(self.P, self.P))
                L_m = _compute_L_from_M(M[m,:,:])
                for i in range(self.P):
                    for j in range(self.P):
                        E = np.zeros(shape=(self.P, self.P))
                        E[i,j] = 1
                        if i == j:
                            grad_sigma_M_m = np.exp(M[m,i,i]) * (E @ L_m.T + L_m @ E)
                        elif i < j:
                            grad_sigma_M_m = (E @ L_m.T + L_m @ E.T)
                        else:
                            pass
                        grad_M[m,i,j] = np.trace(grad_sigma[m,:,:].T @ grad_sigma_M_m)
                        
            # WE WANT TO MAXIMISE THE ELBO, SO APPLY ADAM TO THE NEGATIVE GRADIENT
            grad_M = -grad_M
                            
            # ADAM steps
            first_moment_theta = beta1 * first_moment_theta + (1 - beta1) * grad_theta
            second_moment_theta = beta2 * second_moment_theta + (1 - beta2) * grad_theta ** 2
            first_moment_M = beta1 * first_moment_M + (1 - beta1) * grad_M
            second_moment_M = beta2 * second_moment_M + (1 - beta2) * grad_M ** 2

            first_moment_theta_bias = first_moment_theta / (1 - beta1 ** step)
            second_moment_theta_bias = second_moment_theta / (1 - beta2 ** step)
            first_moment_M_bias = first_moment_M / (1 - beta1 ** step)
            second_moment_M_bias = second_moment_M / (1 - beta2 ** step)
    
            self.theta_phi = (
                self.theta_phi 
                - alpha * first_moment_theta_bias / (np.sqrt(second_moment_theta_bias) + eps)
            )
            M = (
                M 
                - alpha * first_moment_M_bias / (np.sqrt(second_moment_M_bias) + eps)
            ) 
            
            # Convert back to sigma_phi 
            for m in range(self.M_w):
                self.sigma_phi[m,:,:], sigma_phi_inv[m,:,:] = (
                    _compute_sigma_and_inv_from_M(M[m,:,:])
                )
                
            self._compute_ELBO_wrt_phi(num_mc, CAVI_rep, step - 1)
            
    def _compute_ELBO_wrt_phi(self, num_mc, CAVI_rep, step):
        """
        """
        ELBO = np.zeros((self.M_w, ))
                
        # Sample phi and compute delta using current variational parameter values
        phi_k_samps = self._sample_phi(num_mc)
        
        ## Compute the cdf terms    
        einsum_term = np.einsum('np,mpc->nmc', self.features, phi_k_samps, optimize=True) # (num_nodes, num_glob_groups, num_mc)
        cdf_term = np.log(norm.cdf(einsum_term) + 10e-10) # Shift away from 0
        cdf_term_sub = np.log(1 - norm.cdf(einsum_term) + 10e-10) # Shift away from 0
        
        # Average over final axis to MC approximation
        expec_approx = cdf_term.mean(axis=-1)
        expec_approx_sub = cdf_term_sub.mean(axis=-1)
    
        # Multiply by phi_w and sum over nodes
        ELBO += np.einsum('nm,nm->m', self.phi_w, expec_approx)
        ELBO += np.einsum('nm,nm->m',
                                (np.cumsum(self.phi_w[:,::-1], axis=1)[:, ::-1] - self.phi_w), 
                                expec_approx_sub)
        
        ELBO += (0.5 * self.nu_sigma2 / self.omega_sigma2 *
                 (np.trace(self.sigma_phi, axis1=1, axis2=2) + 
                  np.sum(self.theta_phi * self.theta_phi, axis=1) -
                  2 * np.sum(self.theta_phi * self.theta_phi0, axis=1)) +
                 0.5 * np.log(np.linalg.det(self.sigma_phi) + 10e-10)
                 )
        
        self.ELBO_store_phi[CAVI_rep, step, :] = ELBO

    def _update_q_phi0(self):
        """
        A method for computing the variational approximation for each \phi^0_k,
        """
        for k in range(self.M_w):
            self.theta_phi0[k,:] = (
                (self.nu_sigma2[k] * self.theta_phi[k,:] + self.omega_sigma2[k] * self.mu) / 
                (self.nu_sigma2[k] + self.omega_sigma2[k])
            )
            
            self.sigma_phi0[k,:,:] = (
                (self.omega_sigma2[k]) / (self.nu_sigma2[k] + self.omega_sigma2[k]) * np.eye(self.P)
            )
        
    def _compute_ELBO_wrt_phi0(self, CAVI_rep: int, num_mc: int, idx: int):
        """
        """
        def compute_expected_log_joint():
            """
            """
            ELBO_temp = 0
            
            # Term 5: p(\phi | \phi^0, \sigma^2)
            for k in range(self.M_w):
                ELBO_temp += (
                    -0.5 * self.P * (np.log(self.nu_sigma2[k]) - self.omega_sigma2[k]) -
                    0.5 * self.nu_sigma2[k] / self.omega_sigma2[k] * (
                        np.dot(self.theta_phi[k] - self.theta_phi0[k],
                               self.theta_phi[k] - self.theta_phi0[k]) +
                        np.trace(self.sigma_phi[k] + np.trace(self.sigma_phi0[k]))
                    )
                )
                
            # Term 6: p(\phi^0)
            for k in range(self.M_w):
                ELBO_temp += (
                    -0.5 * np.dot(self.theta_phi0[k] - self.mu,
                                  self.theta_phi0[k] - self.mu) -
                    0.5 * np.trace(self.sigma_phi0[k])
                )
 
            return ELBO_temp
        
        def compute_expected_log_q():
            """
            """
            ELBO_temp = 0
            
            # Term 6: q(\phi^0)
            for k in range(self.M_w):
                ELBO_temp += -0.5 * np.log(np.linalg.det(self.sigma_phi0[k]) + 10e-10)
        
            return ELBO_temp
        
        # Compute and store
        self.ELBO_store_phi0[CAVI_rep, idx] = (
            compute_expected_log_joint() - compute_expected_log_q()
        )
        
    def _update_q_sigma2(self):
        """
        A method for computing the variational approximation for each \sigma_k^2.
        """
        self.nu_sigma2 = np.tile(self.nu_0 + self.P / 2, (self.M_w, ))
        self.omega_sigma2 = (
            self.omega_0 + 
            np.sum((self.theta_phi - self.theta_phi0) ** 2, axis=1) / 2 +
            np.trace(self.sigma_phi, axis1=1, axis2=2) / 2 + 
            np.trace(self.sigma_phi0, axis1=1, axis2=2) / 2
        )

    def _compute_ELBO_wrt_sigma2(self, CAVI_rep: int, idx: int):
        """
        """
        def compute_expected_log_joint():
            """
            """
            ELBO_temp = 0

            # Term 5: p(\phi | \phi^0, \sigma^2)
            for k in range(self.M_w):
                ELBO_temp += (
                    -0.5 * self.P * (np.log(self.nu_sigma2[k] + 10e-10) - self.omega_sigma2[k]) -
                    0.5 * self.nu_sigma2[k] / self.omega_sigma2[k] * (
                        np.dot(self.theta_phi[k] - self.theta_phi0[k],
                               self.theta_phi[k] - self.theta_phi0[k]) +
                        np.trace(self.sigma_phi[k] + np.trace(self.sigma_phi0[k]))
                    )
                )
                
            # Term 7: p(\sigma^2)
            for k in range(self.M_w):
                ELBO_temp += (
                    -(self.nu_0 + 1) * (np.log(self.nu_sigma2[k]) - self.omega_sigma2[k]) -
                    self.omega_0 * self.nu_sigma2[k] / self.omega_sigma2[k]
                )
            
            return ELBO_temp
        
        def compute_expected_log_q():
            """
            """
            ELBO_temp = 0
            
            # Term 7: q(\sigma^2)
            ELBO_temp += (
                -(self.nu_sigma2 + 1) * (
                    np.log(self.omega_sigma2 + 10e-10) - 
                    digamma(self.omega_sigma2)
                    )
                - self.nu_sigma2 + self.nu_sigma2 * np.log(self.omega_sigma2 + 10e-10) 
                - loggamma(self.nu_sigma2)
            ).sum()
            
            return ELBO_temp
        
        # Compute and store
        self.ELBO_store_sigma2[CAVI_rep, idx] = (
            compute_expected_log_joint() - compute_expected_log_q()
        )

    def _update_q_gamma(self):
        """
        A method for computing the variational approximation for each \gamma_{rs}.
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
        
        self.alpha_gamma = np.array(term).reshape((self.M_w, self.M_u))

        term = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_beta)(k, s) 
                   for k in range(self.M_w) 
                   for s in range(self.M_u)) 

        self.beta_gamma = np.array(term).reshape((self.M_w, self.M_u))
    
    def _compute_ELBO_wrt_gamma(self, CAVI_rep: int, idx: int):
        """
        """
        def compute_expected_log_joint():
            """
            """
            ELBO_temp = 0
            
            digamma_alpha_gamma = (
                digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma)
                )
            digamma_beta_gamma = (
                digamma(self.beta_gamma) - digamma(self.beta_gamma + self.beta_gamma)
                )
            
            # Term 3: p(z | w, \gamma)
            cumsum_gamma = (
                digamma_alpha_gamma + 
                np.cumsum(digamma_beta_gamma, axis=1) -
                digamma_beta_gamma
                )
                
            ELBO_temp += np.einsum('lik,iw,wk->', self.phi_u, self.phi_w,
                              cumsum_gamma,
                              optimize=True)
            
            # Term 8: p(\gamma')
            ELBO_temp += (self.nu_0 - 1) * digamma_beta_gamma.sum()
            
            return ELBO_temp
        
        def compute_expected_log_q():
            """
            """
            ELBO_temp = 0
            
            digamma_alpha_gamma = (
                digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma)
                )
            digamma_beta_gamma = (
                digamma(self.beta_gamma) - digamma(self.beta_gamma + self.beta_gamma)
                )
            
            # Term 4: q(gamma')
            ELBO_temp += (
                (self.alpha_gamma - 1) * digamma_alpha_gamma + 
                (self.beta_gamma - 1) * digamma_beta_gamma +
                loggamma(self.alpha_gamma + self.beta_gamma) -
                loggamma(self.alpha_gamma) - loggamma(self.beta_gamma)
            ).sum()
            
            return ELBO_temp
        
        # Compute and store
        self.ELBO_store_gamma[CAVI_rep, idx] = (
            compute_expected_log_joint() - compute_expected_log_q()
        )

    def _update_q_rho(self):
        """
        A method for computing the variational approximation for each \rho_{km}.
        """
        def compute_einsum_terms_alpha(l,i):
            adj_tensor_mask = self.adj_tensor[l,i,:].copy()
            adj_tensor_mask[i] = 0   
            
            # w' = x, u' = v
            local_term = np.einsum('j,k,jm->km',
                                   adj_tensor_mask, self.phi_u[l, i, :],
                                   self.phi_u[l, :, :]
            )                                 

            return local_term

        def compute_einsum_terms_beta(l,i):
            adj_tensor_sub_mask = 1 - self.adj_tensor[l,i,:].copy()
            adj_tensor_sub_mask[i] = 0  

            # w' = x, u' = v
            local_term = np.einsum('j,k,jm->km',
                                   adj_tensor_sub_mask, 
                                   self.phi_u[l, i, :],
                                   self.phi_u[l, :, :]
            )   

            return local_term
        
        # Parallel computation 
        term_updates = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_alpha)(l, i) 
                                            for l in range(self.num_layers) 
                                            for i in range(self.num_nodes))

        # Add results back to alpha_rho
        self.alpha_rho = self.alpha_0
        self.alpha_rho += np.array(term_updates).sum(axis=0)
        
        term_updates = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_beta)(l, i) 
                                            for l in range(self.num_layers) 
                                            for i in range(self.num_nodes))

        # Add results back to beta_rho
        self.beta_rho = self.beta_0
        self.beta_rho += np.array(term_updates).sum(axis=0)
        
    def _compute_ELBO_wrt_rho(self, CAVI_rep: int, idx: int):
        """
        """
        def compute_expected_log_joint():
            """
            """
            ELBO_temp = 0
            
            digamma_alpha_rho = np.zeros_like(self.alpha_rho)
            digamma_beta_rho = np.zeros_like(self.beta_rho)
            valid_mask = (self.alpha_rho  > 0) & (self.beta_rho > 0)
            digamma_alpha_rho[valid_mask] = (
                digamma(self.alpha_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            digamma_beta_rho[valid_mask] = (
                digamma(self.beta_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            
            # Term 1: p(A | z, \rho)
            adj_tensor_mask = self.adj_tensor.copy()
            adj_tensor_sub_mask = self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_mask[l,:,:], 0)
                np.fill_diagonal(adj_tensor_sub_mask[l,:,:], 0)
            
            ELBO_temp += np.einsum('lik,ljm,lij,km->', self.phi_u, self.phi_u, 
                              adj_tensor_mask, digamma_alpha_rho,
                              optimize=True)
            
            ELBO_temp += np.einsum('lik,ljm,lij,km->', self.phi_u, self.phi_u, 
                              adj_tensor_sub_mask, digamma_beta_rho,
                              optimize=True)
            
            # Term 2: p(\rho)
            ELBO_temp += (self.alpha_0 - 1) * digamma_alpha_rho.sum() 
            ELBO_temp += (self.beta_0 - 1) * digamma_beta_rho.sum()
            
            return ELBO_temp
        
        def compute_expected_log_q():
            """
            """
            ELBO_temp = 0
            
            digamma_alpha_rho = np.zeros_like(self.alpha_rho)
            digamma_beta_rho = np.zeros_like(self.beta_rho)
            valid_mask = (self.alpha_rho  > 0) & (self.beta_rho > 0)
            digamma_alpha_rho[valid_mask] = (
                digamma(self.alpha_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            digamma_beta_rho[valid_mask] = (
                digamma(self.beta_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            
            # Term 2: q(\rho)
            ELBO_temp += (
                (self.alpha_rho - 1) * digamma_alpha_rho +
                (self.beta_rho - 1) * digamma_beta_rho +
                loggamma(self.alpha_rho + self.beta_rho) -
                loggamma(self.alpha_rho) - loggamma(self.beta_rho)
            ).sum()
            
            return ELBO_temp
        
        # Compute and store
        self.ELBO_store_rho[CAVI_rep, idx] = (
            compute_expected_log_joint() - compute_expected_log_q()
        )
        
    def _compute_full_ELBO(self, CAVI_rep: int, num_mc: int):
        """
        """
        def compute_expected_log_joint():
            """
            """
            ELBO_temp = 0
            
            digamma_alpha_rho = np.zeros_like(self.alpha_rho)
            digamma_beta_rho = np.zeros_like(self.beta_rho)
            valid_mask = (self.alpha_rho  > 0) & (self.beta_rho > 0)
            digamma_alpha_rho[valid_mask] = (
                digamma(self.alpha_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            digamma_beta_rho[valid_mask] = (
                digamma(self.beta_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            
            digamma_alpha_gamma = (
                digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma)
                )
            digamma_beta_gamma = (
                digamma(self.beta_gamma) - digamma(self.beta_gamma + self.beta_gamma)
                )
            
            # Term 1: p(A | z, \rho)
            adj_tensor_mask = self.adj_tensor.copy()
            adj_tensor_sub_mask = self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_mask[l,:,:], 0)
                np.fill_diagonal(adj_tensor_sub_mask[l,:,:], 0)
            
            ELBO_temp += np.einsum('lik,ljm,lij,km->', self.phi_u, self.phi_u, 
                              adj_tensor_mask, digamma_alpha_rho,
                              optimize=True)
            
            ELBO_temp += np.einsum('lik,ljm,lij,km->', self.phi_u, self.phi_u, 
                              adj_tensor_sub_mask, digamma_beta_rho,
                              optimize=True)
            
            # Term 2: p(\rho)
            ELBO_temp += (self.alpha_0 - 1) * digamma_alpha_rho.sum() 
            ELBO_temp += (self.beta_0 - 1) * digamma_beta_rho.sum()
            
            # Term 3: p(z | w, \gamma)
            cumsum_gamma = (
                digamma_alpha_gamma + 
                np.cumsum(digamma_beta_gamma, axis=1) -
                digamma_beta_gamma
                )
                
            ELBO_temp += np.einsum('lik,iw,wk->', self.phi_u, self.phi_w,
                              cumsum_gamma,
                              optimize=True)
            
            # Term 4: p(w | \tau)
            # MC step for the expectation
            tau_ik_samps = self._compute_tau(num_mc)
            log_tau_expec = np.log(tau_ik_samps + 10e-10).mean(axis=2)
            
            ELBO_temp += np.einsum('iw,iw->', self.phi_w, log_tau_expec)
            
            # Term 5: p(\phi | \phi^0, \sigma^2)
            for k in range(self.M_w):
                ELBO_temp += (
                    -0.5 * self.P * (np.log(self.nu_sigma2[k]) - self.omega_sigma2[k]) -
                    0.5 * self.nu_sigma2[k] / self.omega_sigma2[k] * (
                        np.dot(self.theta_phi[k] - self.theta_phi0[k],
                               self.theta_phi[k] - self.theta_phi0[k]) +
                        np.trace(self.sigma_phi[k] + np.trace(self.sigma_phi0[k]))
                    )
                )
                
            # Term 6: p(\phi^0)
            for k in range(self.M_w):
                ELBO_temp += (
                    -0.5 * np.dot(self.theta_phi0[k] - self.mu,
                                  self.theta_phi0[k] - self.mu) -
                    0.5 * np.trace(self.sigma_phi0[k])
                )
                
            # Term 7: p(\sigma^2)
            for k in range(self.M_w):
                ELBO_temp += (
                    -(self.nu_0 + 1) * (np.log(self.nu_sigma2[k]) - self.omega_sigma2[k]) -
                    self.omega_0 * self.nu_sigma2[k] / self.omega_sigma2[k]
                )
            
            # Term 8: p(\gamma')
            ELBO_temp += (self.nu_0 - 1) * digamma_beta_gamma.sum()
            
            return ELBO_temp
        
        def compute_expected_log_q():
            """
            """
            ELBO_temp = 0
            
            digamma_alpha_rho = np.zeros_like(self.alpha_rho)
            digamma_beta_rho = np.zeros_like(self.beta_rho)
            valid_mask = (self.alpha_rho  > 0) & (self.beta_rho > 0)
            digamma_alpha_rho[valid_mask] = (
                digamma(self.alpha_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            digamma_beta_rho[valid_mask] = (
                digamma(self.beta_rho[valid_mask]) 
                - digamma(self.alpha_rho[valid_mask] + self.beta_rho[valid_mask])
            )
            
            digamma_alpha_gamma = (
                digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma)
                )
            digamma_beta_gamma = (
                digamma(self.beta_gamma) - digamma(self.beta_gamma + self.beta_gamma)
                )
            
            # Term 1: q(z)
            ELBO_temp += (self.phi_u * np.log(self.phi_u + 10e-10)).sum()
            
            # Term 2: q(\rho)
            ELBO_temp += (
                (self.alpha_rho - 1) * digamma_alpha_rho +
                (self.beta_rho - 1) * digamma_beta_rho +
                loggamma(self.alpha_rho + self.beta_rho) -
                loggamma(self.alpha_rho) - loggamma(self.beta_rho)
            ).sum()
            
            # Term 3: q(w)
            ELBO_temp += (self.phi_w * np.log(self.phi_w + 10e-10)).sum()
            
            # Term 4: q(gamma')
            ELBO_temp += (
                (self.alpha_gamma - 1) * digamma_alpha_gamma + 
                (self.beta_gamma - 1) * digamma_beta_gamma +
                loggamma(self.alpha_gamma + self.beta_gamma) -
                loggamma(self.alpha_gamma) - loggamma(self.beta_gamma)
            ).sum()
            
            # Term 5: q(\phi)
            for k in range(self.M_w):
                ELBO_temp += -0.5 * np.log(np.linalg.det(self.sigma_phi[k]) + 10e-10)
            
            # Term 6: q(\phi^0)
            for k in range(self.M_w):
                ELBO_temp += -0.5 * np.log(np.linalg.det(self.sigma_phi0[k]) + 10e-10)
                
            # Term 7: q(\sigma^2)
            ELBO_temp += (
                -(self.nu_sigma2 + 1) * (
                    np.log(self.omega_sigma2 + 10e-10) - 
                    digamma(self.omega_sigma2)
                    )
                - self.nu_sigma2 + self.nu_sigma2 * np.log(self.omega_sigma2 + 10e-10) 
                - loggamma(self.nu_sigma2)
            ).sum()
            
            return ELBO_temp
        
        # Compute and store
        ELBO = (
            compute_expected_log_joint() - compute_expected_log_q()
        )
            
        return ELBO
            
    def run_VB_scheme(self, n_CAVI_its: int, num_mc: int, num_grad_steps: int,
                      alpha: float=0.001, beta1: float=0.9, beta2: float=0.999, 
                      eps: float=10**-8):
        """
        Run the full VB update scheme.
        Parameter:
            - n_CAVI_its: number of CAVI iterations.
            - num_mc: number of MC samples for expectation estimates.
            - num_grad_steps: number of gradient ascent steps in the ADAM procedure.
        """
        # Initialise parameter values
        self._initialise_parameters()
        
        # Empty arrays for storing ELBO values
        self.ELBO_store_full = np.zeros((n_CAVI_its * 8, ))
        self.ELBO_store_u = np.zeros((n_CAVI_its, 2))
        self.ELBO_store_w = np.zeros((n_CAVI_its, 2))
        self.ELBO_store_phi = np.zeros((n_CAVI_its, num_grad_steps, self.M_w))
        self.ELBO_store_phi0 = np.zeros((n_CAVI_its, 2))
        self.ELBO_store_sigma2 = np.zeros((n_CAVI_its, 2))
        self.ELBO_store_gamma = np.zeros((n_CAVI_its, 2))
        self.ELBO_store_rho = np.zeros((n_CAVI_its, 2))
        
        # Empty array for tracking estimates
        self.phi_w_store = np.zeros((n_CAVI_its + 1, self.num_nodes, self.M_w))
        self.phi_w_store[0,:,:] = self.phi_w.copy()
        self.phi_u_store = np.zeros((n_CAVI_its + 1, self.num_layers, 
                                     self.num_nodes, self.M_u))
        self.phi_u_store[0,:,:,:] = self.phi_u.copy()
        self.alpha_rho_store = np.zeros((n_CAVI_its + 1, self.M_u, self.M_u))
        self.beta_rho_store = np.zeros((n_CAVI_its + 1, self.M_u, self.M_u))
        self.alpha_rho_store[0] = self.alpha_rho
        self.beta_rho_store[0] = self.beta_rho
    
        for CAVI_rep in range(n_CAVI_its):
            print(f"Iteration {CAVI_rep + 1} of {n_CAVI_its}")

            print("Updating rho")
            self._compute_ELBO_wrt_rho(CAVI_rep, 0)
            self._update_q_rho()
            self._compute_ELBO_wrt_rho(CAVI_rep, 1)
            self._compute_full_ELBO(CAVI_rep, num_mc)
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep] = ELBO_full
            self.alpha_rho_store[CAVI_rep + 1] = self.alpha_rho
            self.beta_rho_store[CAVI_rep + 1] = self.beta_rho
            
            print("Updating gamma")
            self._compute_ELBO_wrt_gamma(CAVI_rep, 0)
            self._update_q_gamma()
            self._compute_ELBO_wrt_gamma(CAVI_rep, 1)
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep + 1] = ELBO_full
        
            print("Updating phi_0")
            self._compute_ELBO_wrt_phi0(CAVI_rep, num_mc, 0)
            self._update_q_phi0()
            self._compute_ELBO_wrt_phi0(CAVI_rep, num_mc, 1)
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep + 2] = ELBO_full
            
            print("Updating phi")
            self._update_q_phi(num_mc, num_grad_steps, CAVI_rep, alpha, beta1,
                               beta2, eps)
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep + 3] = ELBO_full
            
            print("Updating sigma2")
            self._compute_ELBO_wrt_sigma2(CAVI_rep, 0) 
            self._update_q_sigma2()
            self._compute_ELBO_wrt_sigma2(CAVI_rep, 1)
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep + 4] = ELBO_full

            print("Updating u")
            self._compute_ELBO_wrt_u(CAVI_rep, 0)
            self._update_q_u()
            self._compute_ELBO_wrt_u(CAVI_rep, 1)
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep + 5] = ELBO_full
            self.phi_u_store[CAVI_rep + 1,:,:,:] = self.phi_u.copy()

            print("Updating w")
            self._compute_ELBO_wrt_w(CAVI_rep, num_mc, 0)
            self._update_q_w(num_mc)
            self._compute_ELBO_wrt_w(CAVI_rep, num_mc, 1)
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep + 6] = ELBO_full
            self.phi_w_store[CAVI_rep + 1,:,:] = self.phi_w.copy()
            
            print("Computing ELBO")
            ELBO_full = self._compute_full_ELBO(CAVI_rep, num_mc)
            self.ELBO_store_full[8 * CAVI_rep + 7] = ELBO_full
    
