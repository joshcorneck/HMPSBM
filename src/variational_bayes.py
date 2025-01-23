import numpy as np
import math

from scipy.special import gamma, loggamma, digamma, logsumexp
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from scipy.special import comb

import src.helper_functions.initialise_parameters as initialise
from src.helper_functions.compute_ELBO import compute_full_ELBO
from src.helper_functions.phi_gradients import (
    _compute_L_from_M, _compute_M_from_sigma, _compute_sigma_and_inv_from_M,
    compute_gradient_theta_phi_k, compute_gradient_M_phi_k
)
from src.helper_functions.ADAM_methods import (
    ADAM_joint, ADAM_single, ADAM_single_torch
)

class VariationalBayes:

    def __init__(self, num_nodes: int, num_layers: int, adj_tensor: np.array, 
                 features: np.array, M_w: int, M_z: int, 
                 num_fp_its: int=1, k_max: int = 4, num_adj_samps: int=1) -> None:
        """
        A class to compute a mean-field variational approximation to the posterior.
        Parameters:
            - num_nodes: number of nodes in the network (N).
            - num_layers: number of layers in the network (L).
            - adj_tensor: adjacency tensor for the network, of shape (L,N,N).
            - features: features for each node of the network, of shape (N,P).
            - M_w, M_z: values for variational approximation truncation.
            - k_max: embedding dimension used for initialisation.
            - num_adj_samps: how many observations contribute to adj_tensor (M in paper).
        """
        self.num_fp_its = num_fp_its
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.adj_tensor = adj_tensor
        self.features = features
        self.P = features.shape[1]
        self.M_w = M_w; self.M_z = M_z        
        self.k_max = k_max
        self.num_adj_samps = num_adj_samps
            
        self.nu_sigma2 = np.ones((M_w, ))
        self.omega_sigma2 = np.ones((M_w, ))
        self.mu = np.zeros((self.P, ))
        
        self.nu_0 = 1
        self.omega_0 = 1
        self.alpha_0 = 1
        self.beta_0 = 1
        self.eta_0 = 1 / self.M_z
        
        ## ADAM parameters
        # Store the steps for ADAM (so we can continue to next iterations)
        self.step_theta = np.ones((self.M_w), dtype=int)
        self.step_sigma = np.ones_like(self.step_theta)

        self.first_moment_theta_best = np.zeros((self.M_w, self.P))
        self.first_moment_M_best = np.zeros((self.M_w, self.P, self.P))
        self.second_moment_theta_best = np.zeros((self.M_w, self.P))
        self.second_moment_M_best = np.zeros((self.M_w, self.P, self.P))

        # Precompute matrix of M choose A_{\ell ij}
        self.log_M_choose = np.zeros_like(self.adj_tensor, dtype=float)
        for l in range(self.num_layers):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    self.log_M_choose[l,i,j] = np.log(comb(self.num_adj_samps, self.adj_tensor[l,i,j], exact=False))
        
    def _initialise_parameters(self):
        """
        Initialise all parameter values.
        """
        # Initialise phi_u
        self.phi_z = initialise.initialise_phi_u(self.num_nodes, self.num_layers,
                                                 self.k_max, self.M_z, self.adj_tensor)
        
        # Initialise phi_w
        self.phi_w = initialise.initialise_phi_w(self.num_nodes, self.num_layers, 
                                                 self.k_max, self.M_w, self.features, 
                                                 self.adj_tensor)
        
        # Initialise alpha_rho and beta_rho
        self.alpha_rho, self.beta_rho = (
            initialise.initialise_alpha_beta_rho(self.num_layers, self.M_z, self.adj_tensor,
                                                 self.phi_z)
        )
        
        # Initialise alpha_gamma and beta_gamma
        self.alpha_gamma, self.beta_gamma = (
            initialise.initialise_alpha_beta_gamma(self.adj_tensor, self.phi_w,
                                                   self.phi_z)
        )
                
        # Initialise theta_phi, theta_phi0, sigma_phi and sigma_phi0
        self.theta_phi = np.stack(initialise.initialise_theta_phi(self.features, self.phi_w))
        self.theta_phi0 = np.ones((self.M_w, self.P))
        print(f"Initialised theta_phi: {self.theta_phi}")
        
        cov_mat = np.cov(self.features.T)
        self.sigma_phi = np.ones((self.M_w, self.P, self.P))
        self.sigma_phi0 = np.ones((self.M_w, self.P, self.P))
        for m in range(self.M_w):
            self.sigma_phi0[m,:,:] = np.eye(self.P)
            self.sigma_phi[m,:,:] = np.eye(self.P)
            
    
    def _update_q_z(self):
        """
        A method for computing the variational approximation for each z_{\ell i}
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
        def compute_einsum_terms_z(l, i, term):
            # THIS IS TO BE REFORMATTED.
            local_term = term.copy()

            ## For comparison with paper: w'=x, u'=v, r'=s
            # adj_tensor_mask defined to ensure not summing over j=i
            adj_tensor_mask = self.adj_tensor[l, i, :].copy()
            adj_tensor_mask[i] = 0
            log_M_choose_mask = self.log_M_choose[l, i, :].copy()
            log_M_choose_mask[i] = 0
            local_term += np.einsum('jm,j->', 
                                    self.phi_z[l, :, :],
                                    log_M_choose_mask,
                                    optimize=True)
            local_term += np.einsum('jm,j,km->k', 
                                    self.phi_z[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_alpha_rho,
                                    optimize=True)

            adj_tensor_mask = self.adj_tensor[l, :, i].copy()
            adj_tensor_mask[i] = 0
            log_M_choose_mask = self.log_M_choose[l, :, i].copy()
            log_M_choose_mask[i] = 0
            local_term += np.einsum('jm,j->', 
                                    self.phi_z[l, :, :],
                                    log_M_choose_mask,
                                    optimize=True)
            local_term += np.einsum('jm,j,mk->k',
                                    self.phi_z[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_alpha_rho,
                                    optimize=True)

            adj_tensor_mask = self.num_adj_samps - self.adj_tensor[l,i,:].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('jm,j,km->k',
                                    self.phi_z[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_beta_rho,
                                    optimize=True)

            adj_tensor_mask = self.num_adj_samps - self.adj_tensor[l,:,i].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('jm,j,mk->k', 
                                    self.phi_z[l, :, :],
                                    adj_tensor_mask,
                                    precomputed_digamma_beta_rho,
                                    optimize=True)

            return local_term
        
        for CAVI_rep in range(self.num_fp_its):
            for l in range(self.num_layers):
                for i in range(self.num_nodes):
                    phi_temp = compute_einsum_terms_z(l, i, term[l,i,:])
                    self.phi_z[l,i,:] = np.exp(phi_temp - logsumexp(phi_temp))
                    
    def _sample_phi(self, num_mc: int, theta_phi: np.array, sigma_phi: np.array,
                    single: bool=False, epsilon=1e-8):
        """
        Includes noise on the diagonal and a cholesky decomposition for numerical
        stability.
        """
        rng = np.random.default_rng()
        
        if not single:
            phi_k_samps = np.zeros((self.M_w, self.P, num_mc))
            for k in range(self.M_w):
                sigma_phi_reg = sigma_phi[k] + epsilon * np.eye(sigma_phi[k].shape[0])
                L = np.linalg.cholesky(sigma_phi_reg)
                phi_k_samps[k] = theta_phi[k][:, np.newaxis] + L @ rng.normal(size=(L.shape[0], num_mc))
        else:
            sigma_phi_reg = sigma_phi + epsilon * np.eye(sigma_phi.shape[0])
            L = np.linalg.cholesky(sigma_phi_reg)
            phi_k_samps = theta_phi[:, np.newaxis] + L @ rng.normal(size=(L.shape[0], num_mc))
            phi_k_samps = phi_k_samps.T
        
        return phi_k_samps
        
    # def _sample_phi(self, num_mc: int, theta_phi: np.array, sigma_phi: np.array,
    #                 single: bool=False):
    #     """
    #     A method to sample values for phi for the current estimates of
    #     the variationl parameters.
    #     Parameters:
    #         - num_mc: the number of MC samples.
    #     """
    #     if not single:
    #         phi_k_samps = np.zeros((self.M_w, self.P, num_mc))
    #         rng = np.random.default_rng() # Speed improvement with this sampler
    #         for k in range(self.M_w):
    #             phi_k_samps[k,:,:] = rng.multivariate_normal(theta_phi[k,:],
    #                                                         sigma_phi[k,:,:],
    #                                                         size=num_mc).T
    #     else:
    #         rng = np.random.default_rng() # Speed improvement with this sampler
    #         phi_k_samps = rng.multivariate_normal(theta_phi,
    #                                               sigma_phi,
    #                                               size=num_mc)

    #     return phi_k_samps
    
    def _compute_log_tau(self, num_mc: int, phi_k_samps: np.array = None, theta_phi: np.array = None,
                     sigma_phi: np.array = None, return_norm: bool = False):
        """
        """
        ## THIS IS NOW DONE ON A LOG-SCALE
        if phi_k_samps is None:
            phi_k_samps = self._sample_phi(num_mc, theta_phi, sigma_phi)
        
        delta_ik_samps = np.zeros((self.num_nodes, self.M_w, num_mc))
        for i in range(self.num_nodes):
            for k in range(self.M_w):
                delta_ik_samps[i,k,:] = np.dot(self.features[i,:], phi_k_samps[k])
                
        log_tau_ik_samps = np.zeros_like(delta_ik_samps)
        for i in range(self.num_nodes):
            for k in range(self.M_w):
                if k == 0:
                    log_tau_ik_samps[i,k,:] = norm.logcdf(delta_ik_samps[i,k,:])
                else:
                    log_tau_ik_samps[i,k,:] = (
                        norm.logcdf(delta_ik_samps[i,k,:]) +
                        norm.logcdf(-delta_ik_samps[i,:k,:]).sum(axis=0)
                    )
                    
        # Normalise
        log_norm_ik_samps = logsumexp(log_tau_ik_samps, axis=1, keepdims=True)
        log_tau_ik_samps_normalised = log_tau_ik_samps - log_norm_ik_samps
    
        return log_tau_ik_samps_normalised
        
    # def _compute_tau(self, num_mc: int, phi_k_samps: np.array = None, theta_phi: np.array = None,
    #                      sigma_phi: np.array = None, return_norm: bool = False):
    #     """
    #     """
    #     ## THIS IS NOW DONE ON A LOG-SCALE
    #     if phi_k_samps is None:
    #         phi_k_samps = self._sample_phi(num_mc, theta_phi, sigma_phi)
        
    #     delta_ik_samps = np.zeros((self.num_nodes, self.M_w, num_mc))
    #     for i in range(self.num_nodes):
    #         for k in range(self.M_w):
    #             delta_ik_samps[i,k,:] = np.dot(self.features[i,:], phi_k_samps[k])
                
    #     log_tau_ik_samps = np.zeros_like(delta_ik_samps)
    #     for i in range(self.num_nodes):
    #         for k in range(self.M_w):
    #             if k == 0:
    #                 log_tau_ik_samps[i,k,:] = norm.logcdf(delta_ik_samps[i,k,:])
    #             else:
    #                 log_tau_ik_samps[i,k,:] = (
    #                     norm.logcdf(delta_ik_samps[i,k,:]) +
    #                     norm.logcdf(-delta_ik_samps[i,:k,:]).sum(axis=0)
    #                 )
                    
    #     # Normalise
    #     log_norm_ik_samps = logsumexp(log_tau_ik_samps, axis=1, keepdims=True)
    #     norm_ik_samps = np.exp(logsumexp(log_tau_ik_samps, axis=1, keepdims=True))
    #     tau_ik_samps = np.exp(log_tau_ik_samps - log_norm_ik_samps)
        
    #     if return_norm:
    #         return tau_ik_samps, norm_ik_samps 
    #     else:
    #         return tau_ik_samps
    
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
    
        term = np.einsum('liu,wu->iw', self.phi_z, cumsum)

        # MC step for the expectation
        log_tau_ik_samps = self._compute_log_tau(num_mc, theta_phi=self.theta_phi, 
                                                 sigma_phi=self.sigma_phi)
        tau_expec = log_tau_ik_samps.mean(axis=2)
        term += tau_expec
    
        # Take exponential and normalise
        for i in range(self.num_nodes):
            self.phi_w[i,:] = np.exp(term[i,:] - logsumexp(term[i,:]))
        
    def _update_q_phi(self, num_mc: int, max_num_grad_steps: int, CAVI_rep: int,
                    alpha: float=0.001, beta1: float=0.9, beta2: float=0.999, 
                    eps: float=10**-8, eps_ELBO: float=10**-3,
                    alpha_set_theta: list=None, alpha_set_sigma: list=None,
                    max_decrease_count: int=3, ADAM_type: str='joint',
                    lr_theta: float=None, lr_sigma: float=None, multiple_lr: bool=False):
        """
        A method for computing the variational approximation for each \phi_{k}. This
        uses an ADAM optimiser to perform the gradient ascent steps to maximise the ELBO.
        Parameters:
            - num_mc: number of MC samples for expectation approximations.
            - max_num_grad_steps: number of ADAM steps to compute.
            - alpha, beta1, beta2, eps: standard ADAM parameters.
            - eps_ELBO: threshold for deciding on convergence of ELBO.
        """
        if ADAM_type == 'joint':
            # ADAM with joint updates (theta_k and Sigma_k)
            (self.theta_phi, self.sigma_phi, 
            self.step_theta, self.step_sigma,
            self.first_moment_theta_best,
            self.first_moment_M_best,
            self.second_moment_theta_best,
            self.second_moment_M_best) = ADAM_joint(
                theta_phi=self.theta_phi, sigma_phi=self.sigma_phi, 
                phi_w=self.phi_w, M_w=self.M_w, P=self.P, num_mc=num_mc, 
                features=self.features, nu_sigma2=self.nu_sigma2, 
                omega_sigma2=self.omega_sigma2, theta_phi0=self.theta_phi0,
                first_moment_theta_best=self.first_moment_theta_best, 
                first_moment_M_best=self.first_moment_M_best,
                second_moment_theta_best=self.second_moment_theta_best,
                second_moment_M_best=self.second_moment_M_best,
                max_num_grad_steps=max_num_grad_steps,
                max_decrease_count=max_decrease_count, 
                _compute_full_ELBO=self._compute_full_ELBO, 
                _compute_phi_ELBO=self._compute_phi_ELBO, 
                _sample_phi=self._sample_phi, 
                alpha_theta=lr_theta,
                alpha_sigma=lr_sigma,
                alpha_set_theta=alpha_set_theta, 
                alpha_set_sigma=alpha_set_sigma,
                step_theta=self.step_theta, step_sigma=self.step_sigma, 
                beta1=beta1, beta2=beta2, eps=eps,
                multiple_lr=multiple_lr
            )
        elif ADAM_type == 'joint torch':
            pass
        elif ADAM_type == 'single':
            # ADAM with single updates (theta_k then Sigma_k)
            (self.theta_phi, self.sigma_phi, 
            self.step_theta, self.step_sigma,
            self.first_moment_theta_best,
            self.first_moment_M_best,
            self.second_moment_theta_best,
            self.second_moment_M_best) = ADAM_single(
                theta_phi=self.theta_phi, sigma_phi=self.sigma_phi, 
                phi_w=self.phi_w, M_w=self.M_w, P=self.P, num_mc=num_mc, 
                features=self.features, nu_sigma2=self.nu_sigma2, 
                omega_sigma2=self.omega_sigma2, theta_phi0=self.theta_phi0,
                first_moment_theta_best=self.first_moment_theta_best, 
                first_moment_M_best=self.first_moment_M_best,
                second_moment_theta_best=self.second_moment_theta_best,
                second_moment_M_best=self.second_moment_M_best,
                max_num_grad_steps=max_num_grad_steps,
                max_decrease_count=max_decrease_count, 
                _compute_full_ELBO=self._compute_full_ELBO, 
                _compute_phi_ELBO=self._compute_phi_ELBO, 
                _sample_phi=self._sample_phi, 
                alpha_theta=lr_theta, alpha_sigma=lr_sigma,
                alpha_set_theta=alpha_set_theta, 
                alpha_set_sigma=alpha_set_sigma,
                step_theta=self.step_theta, step_sigma=self.step_sigma, 
                beta1=beta1, beta2=beta2, eps=eps,
                multiple_lr=multiple_lr
            )
        elif ADAM_type == 'single torch':
            (self.theta_phi, self.sigma_phi) = ADAM_single_torch(
                theta_phi=self.theta_phi, sigma_phi=self.sigma_phi,
                M_w=self.M_w, P=self.P, max_decrease_count=max_decrease_count,
                max_num_grad_steps=max_num_grad_steps,
                _compute_phi_ELBO=self._compute_phi_ELBO,
                _sample_phi=self._sample_phi,
                num_mc=num_mc, features=self.features, phi_w=self.phi_w,
                nu_sigma2=self.nu_sigma2, omega_sigma2=self.omega_sigma2,
                theta_phi0=self.theta_phi0, lr_theta=lr_theta,
                lr_sigma=lr_sigma
            )
            
         
                        
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
        
    def _update_q_gamma(self):
        """
        A method for computing the variational approximation for each \gamma_{rs}.
        """
        def compute_einsum_terms_alpha(k, s):
            return 1 + np.einsum('li,i->',
                                 self.phi_z[:,:,s],
                                 self.phi_w[:,k])
        
        def compute_einsum_terms_beta(k, s):
            # Mask for sum from r=s+1 to M_u only
            mask_s = np.zeros((self.M_z,))
            mask_s[(s+1):] = 1

            return self.eta_0 + np.einsum('r,lir,i->',
                                 mask_s,
                                 self.phi_z,
                                 self.phi_w[:,k])
        
        term = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_alpha)(k, s) 
                   for k in range(self.M_w) 
                   for s in range(self.M_z)) 
        
        self.alpha_gamma = np.array(term).reshape((self.M_w, self.M_z))

        term = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_beta)(k, s) 
                   for k in range(self.M_w) 
                   for s in range(self.M_z)) 

        self.beta_gamma = np.array(term).reshape((self.M_w, self.M_z))

    def _update_q_rho(self):
        """
        A method for computing the variational approximation for each \rho_{km}.
        """
        def compute_einsum_terms_alpha(l,i):
            adj_tensor_mask = self.adj_tensor[l,i,:].copy()
            adj_tensor_mask[i] = 0   
            
            # w' = x, u' = v
            local_term = np.einsum('j,k,jm->km',
                                   adj_tensor_mask, self.phi_z[l, i, :],
                                   self.phi_z[l, :, :]
            )                                 

            return local_term

        def compute_einsum_terms_beta(l,i):
            adj_tensor_sub_mask = self.num_adj_samps - self.adj_tensor[l,i,:].copy()
            adj_tensor_sub_mask[i] = 0  

            # w' = x, u' = v
            local_term = np.einsum('j,k,jm->km',
                                   adj_tensor_sub_mask, 
                                   self.phi_z[l, i, :],
                                   self.phi_z[l, :, :]
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
        
    def _compute_phi_ELBO(self, theta_phi: np.array, sigma_phi: np.array):
        """
        Compute the ELBO with respect to theta_phi_k or sigma_phi_k.
        """
        ELBO = (
            compute_full_ELBO(ELBO_parameter='phi', _compute_log_tau=self._compute_log_tau,
                              num_layers=self.num_layers, num_nodes=self.num_nodes,
                              num_mc=self.num_mc, adj_tensor=self.adj_tensor,
                              M_u=self.M_z, M_w=self.M_w, P=self.P,
                              alpha_rho=self.alpha_rho, beta_rho=self.beta_rho,
                              alpha_0=self.alpha_0, beta_0=self.beta_0,
                              alpha_gamma=self.alpha_gamma, beta_gamma=self.beta_gamma,
                              phi_u=self.phi_z, phi_w=self.phi_w,
                              nu_sigma2=self.nu_sigma2, omega_sigma2=self.omega_sigma2,
                              nu_0=self.nu_0, omega_0=self.omega_0,
                              theta_phi=theta_phi, sigma_phi=sigma_phi,
                              mu=self.mu, theta_phi0=self.theta_phi0,
                              sigma_phi0=self.sigma_phi0)
        )
            
        return ELBO

    def _compute_full_ELBO(self):
        """
        """
        ELBO = (
            compute_full_ELBO(ELBO_parameter='full', _compute_log_tau=self._compute_log_tau,
                              num_layers=self.num_layers, num_nodes=self.num_nodes,
                              num_mc=self.num_mc, adj_tensor=self.adj_tensor,
                              M_u=self.M_z, M_w=self.M_w, P=self.P,
                              alpha_rho=self.alpha_rho, beta_rho=self.beta_rho,
                              alpha_0=self.alpha_0, beta_0=self.beta_0,
                              alpha_gamma=self.alpha_gamma, beta_gamma=self.beta_gamma,
                              phi_u=self.phi_z, phi_w=self.phi_w,
                              nu_sigma2=self.nu_sigma2, omega_sigma2=self.omega_sigma2,
                              nu_0=self.nu_0, omega_0=self.omega_0,
                              theta_phi=self.theta_phi, sigma_phi=self.sigma_phi,
                              mu=self.mu, theta_phi0=self.theta_phi0,
                              sigma_phi0=self.sigma_phi0)
        )
            
        return ELBO
            
    def run_VB_scheme(self, n_CAVI_its: int, num_mc: int, max_num_grad_steps: int,
                      alpha: float=0.001, beta1: float=0.9, beta2: float=0.999, 
                      eps: float=10**-8, eps_ELBO_phi: float = 10**-5,
                      eps_ELBO_full: float = 10**-5,  
                      max_ELBO_dec: int=5, max_ELBO_steady: int=3, 
                      max_ELBO_phi_dec: int=3,
                      alpha_set_theta: list=None, alpha_set_sigma: list=None,
                      ADAM_type: str='joint', lr_theta: float=None,
                      lr_sigma: float=None, lr_decay: float=0.9, 
                      multiple_lr: bool=False):
        """
        Run the full VB update scheme.
        Parameter:
            - n_CAVI_its: number of CAVI iterations.
            - num_mc: number of MC samples for expectation estimates.
            - max_num_grad_steps: number of gradient ascent steps in the ADAM procedure.
        """
        if ADAM_type not in ['joint', 'joint torch', 'single', 'single torch']:
            raise ValueError("""
                             ADAM_type must be one of:
                             'joint', 'joint torch', 'single', 'single torch'
                             """)
        self.num_mc = num_mc
        
        # Initialise parameter values
        self._initialise_parameters()
        
        # Empty arrays for storing ELBO values
        # self.ELBO_store_full = np.zeros((n_CAVI_its * 7, ))
        self.ELBO_store_full = np.zeros((n_CAVI_its, ))
        self.ELBO_convergence = []

        # Empty array for tracking estimates
        self.phi_w_store = np.zeros((n_CAVI_its + 1, self.num_nodes, self.M_w))
        self.phi_w_store[0,:,:] = self.phi_w.copy()
        self.phi_z_store = np.zeros((n_CAVI_its + 1, self.num_layers, 
                                     self.num_nodes, self.M_z))
        self.phi_z_store[0,:,:,:] = self.phi_z.copy()
        self.alpha_rho_store = np.zeros((n_CAVI_its + 1, self.M_z, self.M_z))
        self.beta_rho_store = np.zeros((n_CAVI_its + 1, self.M_z, self.M_z))
        self.alpha_rho_store[0] = self.alpha_rho
        self.beta_rho_store[0] = self.beta_rho
    
        ELBO_dec_track = 0 # Tracker for ELBO decreasing
        ELBO_steady_track = 0 # Tracker for ELBO staying steady
        ELBO_track = 0 # ELBO value tracker

        for CAVI_rep in range(n_CAVI_its):
            if CAVI_rep > 0:
                lr_theta *= lr_decay
                lr_sigma *= lr_decay
            print(f"...Iteration {CAVI_rep + 1} of {n_CAVI_its}...")

            print("Updating rho", end='\r')
            # print("Updating rho")
            ELBO_full = self._compute_full_ELBO()
            print(f"ELBO before: {ELBO_full}")
            self._update_q_rho()
            self.alpha_rho_store[CAVI_rep + 1] = self.alpha_rho
            self.beta_rho_store[CAVI_rep + 1] = self.beta_rho
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO after: {ELBO_full}")
            
            print("Updating gamma", end='\r')
            # print("Updating gamma")
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO before: {ELBO_full}")
            self._update_q_gamma()
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO after: {ELBO_full}")
        
            print("Updating phi_0", end='\r')
            # print("Updating phi_0")
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO before: {ELBO_full}")
            self._update_q_phi0()
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO after: {ELBO_full}")
            
            print("Updating phi", end='\r')
            # print("Updating phi")
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO before: {ELBO_full}")
            self._update_q_phi(num_mc, max_num_grad_steps, CAVI_rep, alpha, beta1,
                               beta2, eps, eps_ELBO=eps_ELBO_phi, alpha_set_theta=alpha_set_theta,
                               alpha_set_sigma=alpha_set_sigma,
                               max_decrease_count=max_ELBO_phi_dec,
                               ADAM_type=ADAM_type, lr_theta=lr_theta,
                               lr_sigma=lr_sigma, multiple_lr=multiple_lr)
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO after: {ELBO_full}")
            
            print("Updating sigma2", end='\r')
            # print("Updating sigma2")
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO before: {ELBO_full}")
            self._update_q_sigma2()
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO after: {ELBO_full}")

            print("Updating u", end='\r')
            # print("Updating u")
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO before: {ELBO_full}")
            self._update_q_z()
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO after: {ELBO_full}")

            print("Updating w", end='\r')
            # print("Updating w")
            # ELBO_full = self._compute_full_ELBO()
            # print(f"ELBO before: {ELBO_full}")
            self._update_q_w(num_mc)
            ELBO_full = self._compute_full_ELBO()
            print(f"ELBO after: {ELBO_full}")
            
            self.ELBO_store_full[CAVI_rep] = ELBO_full
            self.phi_w_store[CAVI_rep + 1,:,:] = self.phi_w.copy()
            
            if CAVI_rep > 0:
                if ELBO_full < ELBO_track:
                    ELBO_dec_track += 1
                    print(f"ELBO decreased {ELBO_dec_track}", end='\r')

                    if ELBO_dec_track == max_ELBO_dec: 
                        print("Stop.")
                        break
                else:
                    ELBO_dec_track = 0
                    if ((ELBO_full - ELBO_track) / ELBO_track < eps_ELBO_full):
                        ELBO_steady_track += 1
                        # print(f"ELBO steady {ELBO_steady_track}")
                        
                        # # if ELBO_steady_track == max_ELBO_steady:
                        # #     print("ELBO converged.")
                        # #     break
                    else:
                        ELBO_steady_track = 0
            ELBO_track = ELBO_full
