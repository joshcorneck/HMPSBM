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

        # Need to include the MCMC step for \E(\log\tau_{iw})
        #  


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
        term += np.array(parallel_terms).reshape(num_nodes, M_w, -1)

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

        parallel_terms = Parallel(n_jobs=-1)(delayed(compute_einsum_terms)(k, r) 
                        for k in range(M_zeta1) 
                        for r in range(M_zeta1)) 
        term += np.array(parallel_terms).reshape(M_zeta1, M_zeta2, -1)

        self.phi_zeta = term

    def _update_q_delta(self):
        """
        """

    def _update_q_gamma(self):
        """
        """
        def compute_einsum_terms_alpha(k, s):
            return 1 + np.einsum('li,i->',
                                 self.phi_u[:,:,s],
                                 self.phi_w[:,k])
        
        def compute_einsum_terms_beta(k, s):
            # Mask for sum from r=s+1 to M_u only
            mask_s = np.zeros((M_u,))
            mask_s[(s+1):] = 1

            return self.eta_0 + np.einsum('r,lir,i->',
                                 mask_s,
                                 self.phi_u,
                                 self.phi_w[:,k])
        
        term = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_alpha)(k, s) 
                   for k in range(M_gamma1) 
                   for s in range(M_gamma2)) 
        
        self.alpha_gamma = term.reshape((M_gamma1, M_gamma2))

        term = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_beta)(k, s) 
                   for k in range(M_gamma1) 
                   for s in range(M_gamma2)) 

        self.beta_gamma = term.reshape((M_gamma1, M_gamma2))

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
        self.alpha_rho += np.array(term_updates).reshape(M_rho, 
                                                         M_rho, 
                                                         -1).sum(axis=2)
        
        term_updates = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_beta)(l, i) 
                                            for l in range(self.num_layers) 
                                            for i in range(self.num_nodes))

        # Add results back to beta_rho
        self.beta_rho = self.beta_0
        self.beta_rho += np.array(term_updates).reshape(M_rho, 
                                                        M_rho, 
                                                        -1).sum(axis=2)