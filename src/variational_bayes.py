import numpy as np
from scipy.special import digamma, logsumexp


class VariationalBayes:

    def __init__(self, num_nodes, num_layers, adj_tensor) -> None:
        """
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.adj_tensor = adj_tensor
    
    

# class VariationalBayes:

#     def __init__(self, num_nodes, num_layers, adj_tensor) -> None:
#         """
#         """
#         self.num_nodes = num_nodes
#         self.num_layers = num_layers
#         self.adj_tensor = adj_tensor

#     def _compute_q_pi_prime(self, xi_0):
#         """
#         """
#         self.alpha_pi = np.ones((self.M_pi, ))
#         self.beta_pi = np.ones((self.M_pi, )) * xi_0

#         self.alpha_pi += np.einsum('krm->m', self.phi_zeta)
#         self.beta_pi += np.einsum('krm->m', 1 - np.cumsum(self.phi_zeta, axis=2))

#     def _compute_q_gamma_prime(self, eta_0):
#         """
#         """
#         self.alpha_gamma = np.ones((self.M_tau, self.M_gamma))
#         self.beta_gamma = np.ones((self.M_tau, self.M_gamma)) * eta_0

#         self.alpha_gamma += np.einsum('ik,lim->km', self.phi_w, self.phi_u)
#         self.beta_gamma += np.einsum('ik,lim->km', self.phi_w, 1 - np.cumsum(self.phi_u, axis=2))

#     def _compute_q_tau_prime(self):
#         """
#         """
#         self.alpha_tau = np.ones((self.num_nodes, self.M_tau))
#         self.beta_tau = np.ones((self.num_nodes, self.M_tau)) * self.y.reshape((self.num_nodes, 1))

#         self.alpha_tau += self.phi_w

#         self.beta_tau += (1 - np.cumsum(self.phi_w, axis=1))

#     def _compute_q_zeta(self):
#         """
#         """
#         self.phi_zeta = np.zeros((self.M_tau, self.M_gamma, self.M_pi))

#         self.phi_zeta += (
#             digamma(self.alpha_pi) - digamma(self.alpha_pi + self.beta_pi) +
#             (np.cumsum(digamma(self.alpha_pi) - 
#                        digamma(self.alpha_pi + self.beta_pi)) - 
#             digamma(self.alpha_pi) - digamma(self.alpha_pi + self.beta_pi))
#             )
        
#         self.phi_zeta += np.einsum('ik,jm,lir,ljs,msq,lij,pq->krp',
#                                    self.phi_w, self.phi_w, self.phi_u, self.phi_u,
#                                    self.phi_zeta, self.adj_tensor,
#                                    digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_zeta += np.einsum('ik,jm,lir,ljs,msq,lij,pq->krp',
#                                    self.phi_w, self.phi_w, self.phi_u, self.phi_u,
#                                    self.phi_zeta, 1 - self.adj_tensor,
#                                    digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_zeta += np.einsum('ik,jm,lir,ljs,msq,lji,qp->krp',
#                                    self.phi_w, self.phi_w, self.phi_u, self.phi_u,
#                                    self.phi_zeta, self.adj_tensor,
#                                    digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_zeta += np.einsum('ik,jm,lir,ljs,msq,lji,qp->krp',
#                                    self.phi_w, self.phi_w, self.phi_u, self.phi_u,
#                                    self.phi_zeta, 1 - self.adj_tensor,
#                                    digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_zeta -= np.einsum('ik,jk,lir,ljr,lij,pp->krp',
#                                    self.phi_w, self.phi_w, self.phi_u, self.phi_u, self.adj_tensor, 
#                                    digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_zeta -= np.einsum('ik,jk,lir,ljr,lij,pp->krp',
#                                    self.phi_w, self.phi_w, self.phi_u, self.phi_u, 1 - self.adj_tensor, 
#                                    digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))

#         logsumexp_phi_zeta = logsumexp(self.phi_zeta, axis=2, keepdims=True)
#         self.phi_zeta = np.exp(self.phi_zeta - logsumexp_phi_zeta)

#     def _compute_q_w(self):
#         """
#         """
#         self.phi_w = np.zeros((self.num_nodes, self.M_tau))

#         self.phi_w += (
#             digamma(self.alpha_tau) - digamma(self.alpha_tau + self.beta_tau) +
#             (np.cumsum(digamma(self.beta_tau) -
#                        digamma(self.alpha_tau + self.beta_tau), axis=1) - 
#             digamma(self.beta_tau) - digamma(self.alpha_tau + self.beta_tau))
#         )

#         self.phi_w += np.einsum('jm,lir,ljs,rka,msb,lij,ab->ik',
#                                 self.phi_w, self.phi_u, self.phi_u, self.phi_zeta, self.phi_zeta,
#                                 self.adj_tensor, digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_w += np.einsum('jm,lir,ljs,rka,msb,lij,ab->ik',
#                                 self.phi_w, self.phi_u, self.phi_u, self.phi_zeta, self.phi_zeta,
#                                 1 - self.adj_tensor, digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_w += np.einsum('jm,lir,ljs,rka,msb,lji,ba->ik',
#                                 self.phi_w, self.phi_u, self.phi_u, self.phi_zeta, self.phi_zeta,
#                                 self.adj_tensor, digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_w += np.einsum('jm,lir,ljs,rka,msb,lji,ba->ik',
#                                 self.phi_w, self.phi_u, self.phi_u, self.phi_zeta, self.phi_zeta,
#                                 1 - self.adj_tensor, digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_w -= np.einsum('lir,kra,lii,aa->ik',
#                                 self.phi_u, self.phi_zeta, self.adj_tensor, 
#                                 digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_w -= np.einsum('lir,kra,lii,aa->ik',
#                                 self.phi_u, self.phi_zeta, 1 - self.adj_tensor, 
#                                 digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
        
#         second_term = (np.cumsum(digamma(self.beta_gamma) -
#                        digamma(self.alpha_gamma + self.beta_gamma), axis=1) - 
#                     digamma(self.beta_gamma) - digamma(self.alpha_gamma + self.beta_gamma))
#         self.phi_w += np.einsum('lim,km->ik', self.phi_u, 
#                                 digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma) +
#                                 second_term)
        
#         logsumexp_phi_w = logsumexp(self.phi_w, axis=1, keepdims=True)
#         self.phi_w = np.exp(self.phi_w - logsumexp_phi_w)

#     def _compute_q_u(self):
#         """
#         """
#         self.phi_u = np.zeros((self.num_layers, self.num_nodes, self.M_gamma))

#         self.phi_u += (
#             digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma) +
#             (np.cumsum(digamma(self.alpha_gamma) - 
#                        digamma(self.alpha_gamma + self.beta_gamma), axis=1) - 
#             digamma(self.alpha_gamma) - digamma(self.alpha_gamma + self.beta_gamma))
#             ).sum(axis=0)
            
#         self.phi_u += np.einsum('ik,jm,ljs,krp,msq,lij,pq->lir',self.phi_w, self.phi_w, self.phi_u, 
#                                 self.phi_zeta, self.phi_zeta, self.adj_tensor,
#                                 digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_u += np.einsum('ik,jm,ljs,krp,msq,lij,pq->lir',self.phi_w, self.phi_w, self.phi_u, 
#                                 self.phi_zeta, self.phi_zeta, 1 - self.adj_tensor,
#                                 digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_u += np.einsum('ik,jm,ljs,krp,msq,lji,qp->lir',self.phi_w, self.phi_w, self.phi_u, 
#                                 self.phi_zeta, self.phi_zeta, self.adj_tensor,
#                                 digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_u += np.einsum('ik,jm,ljs,krp,msq,lji,qp->lir',self.phi_w, self.phi_w, self.phi_u, 
#                                 self.phi_zeta, self.phi_zeta, 1 - self.adj_tensor,
#                                 digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_u -= np.einsum('ik,krp,lii,pp->lir', self.phi_w, self.phi_zeta, self.adj_tensor,
#                                 digamma(self.alpha_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
#         self.phi_u -= np.einsum('ik,krp,lii,pp->lir', self.phi_w, self.phi_zeta, 1 - self.adj_tensor,
#                                 digamma(self.beta_sigma) - digamma(self.alpha_sigma + self.beta_sigma))
        
#         logsumexp_phi_u = logsumexp(self.phi_u, axis=2, keepdims=True)
#         self.phi_u = np.exp(self.phi_u - logsumexp_phi_u)
        
#     def _compute_q_sigma(self, alpha_0, beta_0):
#         """
#         """
#         self.alpha_sigma = np.ones((self.M_pi, self.M_pi)) * alpha_0
#         self.beta_sigma = np.ones((self.M_pi, self.M_pi)) * beta_0

#         self.alpha_sigma += np.einsum('ik,jm,lir,ljs,kra,msb,lij->ab',
#                                       self.phi_w, self.phi_w, self.phi_u, self.phi_u,
#                                       self.phi_zeta, self.phi_zeta, self.adj_tensor)
#         self.beta_sigma += np.einsum('ik,jm,lir,ljs,kra,msb,lij->ab',
#                                       self.phi_w, self.phi_w, self.phi_u, self.phi_u,
#                                       self.phi_zeta, self.phi_zeta, 1 - self.adj_tensor)

#     def run_VB(self, M_tau, M_gamma, M_pi, n_cavi, xi_0, eta_0, alpha_0, beta_0, cov_vec, param_vec):
#         """
#         """
#         self.M_tau = M_tau
#         self.M_gamma = M_gamma
#         self.M_pi = M_pi
#         self.y = np.exp(cov_vec @ param_vec)

#         self.alpha_pi = np.random.uniform(size=(self.M_pi, ))
#         self.beta_pi = np.random.uniform(size=(self.M_pi, )) 

#         self.alpha_gamma = np.random.uniform(size=(self.M_tau, self.M_gamma))
#         self.beta_gamma = np.random.uniform(size=(self.M_tau, self.M_gamma)) 

#         self.alpha_tau = np.random.uniform(size=(self.num_nodes, self.M_tau))
#         self.beta_tau = np.random.uniform(size=(self.num_nodes, self.M_tau)) 

#         self.phi_zeta = np.random.uniform(size=(self.M_tau, self.M_gamma, self.M_pi))
#         self.phi_zeta /= self.phi_zeta.sum(axis=2, keepdims=True)

#         self.phi_w = np.random.uniform(size=(self.num_nodes, self.M_tau))
#         self.phi_w /= self.phi_w.sum(axis=1, keepdims=True)

#         self.phi_u = np.random.uniform(size=(self.num_layers, self.num_nodes, self.M_gamma))
#         self.phi_u /= self.phi_u.sum(axis=2, keepdims=True)

#         self.alpha_sigma = np.random.uniform(size=(self.M_pi, self.M_pi))
#         self.beta_sigma = np.random.uniform(size=(self.M_pi, self.M_pi)) 
        
#         for it in range(n_cavi):
#             self._compute_q_pi_prime(xi_0)
#             self._compute_q_gamma_prime(eta_0)
#             self._compute_q_tau_prime()
#             self._compute_q_zeta()
#             self._compute_q_w()
#             self._compute_q_u()
#             self._compute_q_sigma(alpha_0, beta_0)
