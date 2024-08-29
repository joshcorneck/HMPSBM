import numpy as np

from scipy.special import digamma, logsumexp
from joblib import Parallel, delayed

class VariationalBayes:

    def __init__(self, num_nodes, num_layers, adj_tensor, features) -> None:
        """
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.adj_tensor = adj_tensor
        self.features = features
    
        # Initialise empty arrays for the variational parameters.
        self.phi_u = np.zeros((num_layers, num_nodes, M_u))
        self.phi_w = np.zeros((num_nodes, M_w))
        self.phi_zeta = np.zeros((M_zeta1, M_zeta2, M_zeta3))
        self.theta_delta = np.zeros((num_nodes, M_delta))
        self.sigma_delta = np.zeros((num_nodes, M_delta))
        self.theta_phi = None
        self.sigma_phi = None
        self.alpha_gamma = np.zeros((M_gamma1, M_gamma2))
        self.beta_gamma = np.zeros((M_gamma1, M_gamma2))
        self.alpha_pi = np.zeros((M_pi, ))
        self.beta_pi = np.zeros((M_pi, ))
        self.alpha_rho = np.zeros((M_rho, M_rho))
        self.beta_rho = np.zeros((M_rho, M_rho))

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
        term = self.phi_w @ (digamma(self.alpha_gamma) - 
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
        term += self.phi_w @ cumsum
        
        term = np.tile(term, (self.num_layers, 1, 1))
        
        # Function for paraellelising the computations
        def compute_einsum_terms(l, i):
            local_term = np.zeros(M_u)

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

    def _update_q_w(self):
        """
        """
        pass

    def _update_q_zeta(self):
        """
        """
        term = (
            digamma(self.alpha_pi) - digamma(self.beta_pi) +
            np.cumsum(digamma(self.beta_pi) - 
                      digamma(self.alpha_pi + self.beta_pi))
        )
        term = np.tile(term, (M_zeta1, M_zeta2, M_zeta3)) # !! M_zeta3 = M_pi !!

        def compute_einsum_terms(k, r):
            """
            """
            # Masks for ensuring we don't sum over unintended indices
            mask_k = np.ones((M_w,)) 
            mask_k[k] = 0
            mask_r = np.ones((M_u,)) 
            mask_r[r] = 0
            adj_tensor_mask = self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_mask[l], 0)
            adj_tensor_sub_mask = 1 - self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_sub_mask[l], 0)

            term = np.zeros(M_zeta3) 

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

        term = Parallel(n_jobs=1)(delayed(compute_einsum_terms)(k, r) 
                        for k in range(M_zeta1) 
                        for r in range(M_zeta1)) 
        term += np.array(term).reshape(M_zeta1, M_zeta2, -1)

        self.phi_zeta = term

    def _update_q_delta(self):
        """
        """

    def _update_q_gamma(self):
        """
        """
        
