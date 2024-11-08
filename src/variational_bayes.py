import numpy as np

from scipy.special import digamma, logsumexp
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.linalg import svd
from sklearn.cluster import DBSCAN, KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import time
import hdbscan

class VariationalBayes:

    def __init__(self, num_nodes: int, num_layers: int, adj_tensor: np.array, 
                 features: np.array, M_w: int, M_u: int, M_zeta: int,
                 num_fp_its: int=1) -> None:
        """
        A class to compute a mean-field variational approximation to the posterior.
        Parameters:
            - num_nodes: number of nodes in the network (N).
            - num_layers: number of layers in the network (L).
            - adj_tensor: adjacency tensor for the network, of shape (L,N,N).
            - features: features for each node of the network, of shape (N,P).
            - M_w, M_u, M_zeta: values for variational approximation truncation.
        """
        self.num_fp_its = num_fp_its
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.adj_tensor = adj_tensor
        self.features = features
        self.P = features.shape[1]
        self.M_w = M_w; self.M_u = M_u; self.M_zeta = M_zeta
    
        # Initialise empty arrays for the variational parameters.
        self.phi_u = np.random.uniform(size=(num_layers, num_nodes, M_u))
        # self.phi_u /= self.phi_u.sum(axis=2, keepdims=True)
        
        self.phi_w = np.random.uniform(size=(num_nodes, M_w))
        # self.phi_w /= self.phi_w.sum(axis=1, keepdims=True)
        
        self.phi_zeta = np.random.uniform(size=(M_w, M_u, M_zeta))
        self.phi_zeta /= self.phi_zeta.sum(axis=2, keepdims=True)
        
        self.theta_phi0 = np.zeros((M_w, self.P))
        self.sigma_phi0 = np.ones((M_w, self.P, self.P))
        
        self.theta_phi = np.zeros((M_w, self.P))
        self.sigma_phi = np.ones((M_w, self.P, self.P))
        
        cov_mat = np.cov(self.features.T)
        for m in range(M_w):
            self.sigma_phi0[m,:,:] = cov_mat
            self.sigma_phi[m,:,:] = cov_mat
            
        self.nu_sigma2 = np.ones((M_w, ))
        self.omega_sigma2 = np.ones((M_w, ))
        self.mu = np.zeros((self.P, ))
        
        self.nu_0 = 1
        self.omega_0 = 1
        self.alpha_0 = 1
        self.beta_0 = 1
        self.xi_0 = 1
        self.eta_0 = 1
        
        self.alpha_gamma = np.random.uniform(size=(M_w, M_u))
        self.beta_gamma = np.random.uniform(size=(M_w, M_u))
        self.alpha_pi = np.random.uniform(size=(M_zeta, ))
        self.beta_pi = np.random.uniform(size=(M_zeta, ))
        self.alpha_rho = np.random.uniform(size=(M_zeta, M_zeta))
        self.beta_rho = np.random.uniform(size=(M_zeta, M_zeta))
        
    def _initialise_parameters(self):
        """
        """
        # Embedding dimension
        k_max = 4

        # Store the embeddings
        left_embed = np.zeros((self.um_nodes, self.num_layers, k_max))

        # Layer clusterings
        layer_left_clusterings = np.zeros((self.num_nodes, self.num_layers))

        def assign_clusters(X, M):
            """
            """
            # Compute the clustering (ensure <= M_u labels)
            hdb = hdbscan.HDBSCAN()
            clustering = hdb.fit(X)
            labels = clustering.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # Cut the tree to obtain M clusters
            while num_clusters > M:
                # Increase the minimum cluster size to merge smaller clusters
                hdb.min_cluster_size += 1
                clustering = hdb.fit(X)
                labels = clustering.labels_
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            return num_clusters, clustering.labels_

        def assign_outliers(num_outliers, clustering, X, num_clusters, 
                            k_max):
            """
            """
            # Assign those with a -1 label to the group to whose centroid the
            # outlier is cloest to
            cluster_centroids = np.zeros((num_clusters, k_max))
            l2_distances = np.zeros((num_outliers, num_clusters))
            for cluster in range(num_clusters):
                cluster_idxs = clustering == cluster
                cluster_centroid = X[cluster_idxs, :].mean(axis=0)
                cluster_centroids[cluster, :] = cluster_centroid 
                l2_distances[:,cluster] = np.linalg.norm(
                    X[outlier_idxs,:] - cluster_centroid, axis=1
                )
            outlier_assignments = np.argmin(l2_distances, axis=1)
            
            return outlier_assignments

        # Embed each layer
        for l in range(self.num_layers):
            A_l = self.adj_tensor[l,:,:]
            U, Sigma, Vt = svd(A_l)
            # Left and right eigenvectors
            left_evecs = U[:,:k_max]; right_evecs = Vt.T[:,:k_max]
            #Compute the embeddings
            Sigma_half = np.sqrt(Sigma)
            X = left_evecs @ np.diag(Sigma_half[:k_max])
            X_prime = right_evecs @ np.diag(Sigma_half[:k_max])
            # Store
            left_embed[:,l,:] = X
            
            # Compute the left_embedding clustering (ensure <= M_u labels)
            num_clusters, layer_left_clusterings[:,l] = assign_clusters(X, self.M_u)
            
            # Assign those with a -1
            outlier_idxs = layer_left_clusterings[:,l] == -1
            if outlier_idxs.sum() != 0:
                layer_left_clusterings[outlier_idxs, l] = (
                    assign_outliers(outlier_idxs.sum(), layer_left_clusterings[:,l],
                                    X, num_clusters, k_max)
                )
            
            # Initiliase phi_u
            self.phi_u[l,np.arange(self.num_nodes),
                       layer_left_clusterings[:,l].astype(int)] = 1
            
        # Embed the average adjacency matrix
        A_avg = self.adj_tensor.mean(axis=0)
        U, Sigma, Vt = svd(A_avg)
        left_evecs = U[:,:k_max]; right_evecs = Vt.T[:,:k_max]
        X = left_evecs @ np.diag(Sigma_half[:k_max])
        X_prime = right_evecs @ np.diag(Sigma_half[:k_max])

        # Compute the right_embedding clustering (ensure <= M_u labels)
        num_clusters, global_left_clusterings = assign_clusters(X, self.M_w)

        # Assign those with a -1 label
        outlier_idxs = global_left_clusterings == -1
        if outlier_idxs.sum() != 0:
            global_left_clusterings[outlier_idxs] = (
                assign_outliers(outlier_idxs.sum(), global_left_clusterings,
                                X, num_clusters, k_max)
            )
            
        self.phi_w[np.arange(self.num_nodes), global_left_clusterings] = 1

    def _update_q_u(self):
        """
        A method for computing the variational approximation for each u_{\ell i}
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
        def compute_einsum_terms_u(l, i):
            # THIS IS TO BE REFORMATTED.
            local_term = term[l,i,:]

            ## For comparison with paper: w'=x, u'=v, r'=s
            # adj_tensor_mask defined to ensure not summing over j=i
            adj_tensor_mask = self.adj_tensor[l, i, :].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('w,jx,jv,wur,xvs,j,rs->u',
                                    self.phi_w[i, :], self.phi_w, 
                                    self.phi_u[l, :, :],
                                    self.phi_zeta, self.phi_zeta,
                                    adj_tensor_mask,
                                    precomputed_digamma_alpha_rho,
                                    optimize=True)

            adj_tensor_mask = self.adj_tensor[l, :, i].copy()
            adj_tensor_mask[i] = 0
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

            adj_tensor_mask = 1 - self.adj_tensor[l,:,i].copy()
            adj_tensor_mask[i] = 0
            local_term += np.einsum('w,jx,jv,wur,xvs,j,sr->u',
                                    self.phi_w[i, :], self.phi_w, 
                                    self.phi_u[l, :, :],
                                    self.phi_zeta, self.phi_zeta,
                                    adj_tensor_mask,
                                    precomputed_digamma_beta_rho,
                                    optimize=True)

            return local_term
        
        # Parallel computation 
        # term_updates = Parallel(n_jobs=-1)(delayed(compute_einsum_terms)(l, i) 
        #                                     for l in range(self.num_layers) 
        #                                     for i in range(self.num_nodes))

        # # Add results back to phi_u
        # term += np.array(term_updates).reshape(self.num_layers, 
        #                                        self.num_nodes, 
        #                                        -1)
        for CAVI_rep in range(self.num_fp_its):
            start = time.time()
            for l in range(self.num_layers):
                for i in range(self.num_nodes):
                    phi_temp = compute_einsum_terms_u(l,i)
                    self.phi_u[l,i,:] = np.exp(phi_temp - logsumexp(phi_temp))
            end = time.time()
            print(f"phi_u: paralellised CAVI rep {CAVI_rep}: {end - start}")
        
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
    
    def _update_q_w(self, num_mc: int):
        """
        A method for computing the variational approximation for each w_i.
        Parameters:
            - num_mc: the number of MC samples used for the MC approximation to the 
                      expectation of delta.
        """
        cumsum = (
            np.cumsum(digamma(self.beta_gamma) -
                      digamma(self.alpha_gamma + 
                              self.beta_gamma), axis=1) -
            (digamma(self.beta_gamma) -
             digamma(self.alpha_gamma +
                     self.beta_gamma))
        )

        term = np.einsum('liu,wu->iw', self.phi_u, cumsum)

        # MC step for the expectation
        delta_ik_samps = self._compute_delta(num_mc)
        self.delta_ik_samps = delta_ik_samps

        # Compute the approximation of the expectation using the samples
        tau = np.zeros((self.num_nodes, self.M_w, num_mc))
        tau[:,1:,:] = (1 - norm.cdf(delta_ik_samps[:,:-1,:]).cumprod(axis=1))
        tau *= norm.cdf(delta_ik_samps)
        tau += 10e-10 # For numerical stability

        term += np.log(tau).mean(axis=2)

        def compute_einsum_terms_w(i):
            """
            """
            # THIS IS TO BE REFORMATTED
            local_term = term[i,:]
            
            adj_tensor_mask = self.adj_tensor[:, i, :].copy()
            adj_tensor_mask[:, i] = 0
            adj_tensor_sub_mask = 1 - self.adj_tensor[:, i, :].copy()
            adj_tensor_sub_mask[:, i] = 0

            # w'=x, u'=v, r'=s
            local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
                                   self.phi_w, self.phi_u[:,i,:], self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_mask, 
                                   digamma(self.alpha_rho) - 
                                   digamma(self.alpha_rho + self.beta_rho),
                                   optimize=True)
            local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
                                   self.phi_w, self.phi_u[:,i,:], self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_sub_mask, 
                                   digamma(self.beta_rho) - 
                                   digamma(self.alpha_rho + self.beta_rho),
                                   optimize=True)
            
            adj_tensor_mask = self.adj_tensor[:, :, i].copy()
            adj_tensor_mask[:, i] = 0
            adj_tensor_sub_mask = 1 - self.adj_tensor[:, :, i].copy()
            adj_tensor_sub_mask[:, i] = 0
            local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,sr->w',
                                   self.phi_w, self.phi_u[:,i,:], self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_mask, 
                                   digamma(self.alpha_rho) - 
                                   digamma(self.alpha_rho + self.beta_rho),
                                   optimize=True)
            local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,sr->w',
                                   self.phi_w, self.phi_u[:,i,:], self.phi_u,
                                   self.phi_zeta, self.phi_zeta,
                                   adj_tensor_sub_mask, 
                                   digamma(self.beta_rho) - 
                                   digamma(self.alpha_rho + self.beta_rho),
                                   optimize=True)
            
            return local_term
        
        # def compute_einsum_terms_w_temp(i):
        #     """
        #     Compute the einsum terms and normalize for a given node i.
        #     """
        #     local_term = term[i, :].copy()

        #     adj_tensor_mask = self.adj_tensor[:, i, :].copy()
        #     adj_tensor_mask[:, i] = 0
        #     adj_tensor_sub_mask = 1 - self.adj_tensor[:, i, :].copy()
        #     adj_tensor_sub_mask[:, i] = 0

        #     # Compute the local term using einsum operations
        #     local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
        #                             self.phi_w, self.phi_u[:, i, :], self.phi_u,
        #                             self.phi_zeta, self.phi_zeta,
        #                             adj_tensor_mask,
        #                             digamma(self.alpha_rho) - 
        #                             digamma(self.alpha_rho + self.beta_rho),
        #                             optimize=True)
        #     local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,rs->w',
        #                             self.phi_w, self.phi_u[:, i, :], self.phi_u,
        #                             self.phi_zeta, self.phi_zeta,
        #                             adj_tensor_sub_mask,
        #                             digamma(self.beta_rho) - 
        #                             digamma(self.alpha_rho + self.beta_rho),
        #                             optimize=True)

        #     adj_tensor_mask = self.adj_tensor[:, :, i].copy()
        #     adj_tensor_mask[:, i] = 0
        #     adj_tensor_sub_mask = 1 - self.adj_tensor[:, :, i].copy()
        #     adj_tensor_sub_mask[:, i] = 0

        #     local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,sr->w',
        #                             self.phi_w, self.phi_u[:, i, :], self.phi_u,
        #                             self.phi_zeta, self.phi_zeta,
        #                             adj_tensor_mask,
        #                             digamma(self.alpha_rho) - 
        #                             digamma(self.alpha_rho + self.beta_rho),
        #                             optimize=True)
        #     local_term += np.einsum('jx,lu,ljv,wur,xvs,lj,sr->w',
        #                             self.phi_w, self.phi_u[:, i, :], self.phi_u,
        #                             self.phi_zeta, self.phi_zeta,
        #                             adj_tensor_sub_mask,
        #                             digamma(self.beta_rho) - 
        #                             digamma(self.alpha_rho + self.beta_rho),
        #                             optimize=True)

        #     # Normalize the local term
        #     phi_temp_normalized = np.exp(local_term - logsumexp(local_term))

        #     return i, phi_temp_normalized   

        # for CAVI_rep in range(self.num_fp_its):
        #     start = time.time()
        #     results = Parallel(n_jobs=-1)(delayed(compute_einsum_terms_w_temp)(i) 
        #                               for i in range(self.num_nodes))
        #     for i, phi_temp_normalized in results:
        #         self.phi_w[i, :] = phi_temp_normalized
        #     end = time.time()
        #     print(f"phi_w: paralellised CAVI rep {CAVI_rep}: {end - start}")
            
        for CAVI_rep in range(self.num_fp_its):
            start = time.time()
            for i in range(self.num_nodes):
                phi_temp = compute_einsum_terms_w(i)
                self.phi_w[i,:] = np.exp(phi_temp - logsumexp(phi_temp))
            end = time.time()
            print(f"Non-paralellised CAVI rep: {end - start}")
        
    def _update_q_zeta(self):
        """
        A method for computing the variational approximation for each \zeta_{k,r}.
        """
        term = (
            np.cumsum(digamma(self.beta_pi) - 
                      digamma(self.alpha_pi + self.beta_pi)) - 
            digamma(self.alpha_pi) - digamma(self.beta_pi)
        )
        term = np.tile(term, (self.M_w, self.M_u, 1))

        def compute_einsum_terms_zeta(k, r):
            """
            """
            # Add the \psi(\alpha_pi) terms
            local_term = term[k, r, :]
                        
            # Masks for ensuring that we don't sum over unintended indices
            mask = np.ones((self.M_w, self.M_u))
            mask[k, r] = 0

            adj_tensor_mask = self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_mask[l], 0)
            adj_tensor_sub_mask = 1 - self.adj_tensor.copy()
            for l in range(self.num_layers):
                np.fill_diagonal(adj_tensor_sub_mask[l], 0)

            temp_term = np.einsum('i,j,li,lj,lij->', self.phi_w[:,k],
                                    self.phi_w[:,k], self.phi_u[:,:,r],
                                    self.phi_u[:,:,r], adj_tensor_mask, 
                                    optimize=True)
            
            temp_term = temp_term * np.diag(
                digamma(self.alpha_rho) - 
                digamma(self.alpha_rho + self.beta_rho)
            )
            
            local_term += temp_term

            temp_term = np.einsum('i,j,li,lj,lij->', self.phi_w[:,k],
                                    self.phi_w[:,k], self.phi_u[:,:,r],
                                    self.phi_u[:,:,r], adj_tensor_sub_mask, 
                                    optimize=True)
            temp_term = temp_term * np.diag(
                digamma(self.beta_rho) - 
                digamma(self.alpha_rho + self.beta_rho)
            )
            
            local_term += temp_term

            # s' = t
            local_term += np.einsum('i,jw,li,lju,wut,wu,lij,st->s',
                            self.phi_w[:,k], self.phi_w,
                            self.phi_u[:,:,r], self.phi_u,
                            self.phi_zeta, mask, adj_tensor_mask,
                            digamma(self.alpha_rho) - 
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)    

            local_term += np.einsum('i,jw,li,lju,wut,wu,lij,st->s',
                            self.phi_w[:,k], self.phi_w,
                            self.phi_u[:,:,r], self.phi_u,
                            self.phi_zeta, mask, adj_tensor_sub_mask,
                            digamma(self.beta_rho) - 
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)
            
            local_term += np.einsum('i,jw,li,lju,wut,wu,lji,ts->s', 
                            self.phi_w[:,k], self.phi_w,
                            self.phi_u[:,:,r], self.phi_u,
                            self.phi_zeta, mask, adj_tensor_mask,
                            digamma(self.alpha_rho) - 
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)
            
            local_term += np.einsum('i,jw,li,lju,wut,wu,lji,ts->s', 
                            self.phi_w[:,k], self.phi_w,
                            self.phi_u[:,:,r], self.phi_u,
                            self.phi_zeta, mask, adj_tensor_sub_mask,
                            digamma(self.beta_rho) - 
                            digamma(self.alpha_rho + self.beta_rho),
                            optimize=True)

            return local_term

        # parallel_terms = Parallel(n_jobs=-1)(delayed(compute_einsum_terms)(k, r) 
        #                 for k in range(self.M_w) 
        #                 for r in range(self.M_u)) 
        # term += np.array(parallel_terms).reshape(self.M_w, self.M_u, self.M_zeta)
    
        # self.phi_zeta = term / term.sum(axis=2, keepdims=True)
        
        for CAVI_rep in range(self.num_fp_its):
            start = time.time()
            for k in range(self.M_w):
                for r in range(self.M_u):
                    phi_temp = compute_einsum_terms_zeta(k,r)
                    self.phi_zeta[k,r,:] = np.exp(phi_temp - logsumexp(phi_temp))
            end = time.time()
            print(f"phi_zeta: paralellised CAVI rep {CAVI_rep}: {end - start}")


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
        
        self.ELBO_store[CAVI_rep, step, :] = ELBO

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

    def _update_q_pi(self):
        """
        A method for computing the variational approximation for \pi.
        """
        self.alpha_pi = 1 + np.einsum('krs->s', self.phi_zeta)
        beta_temp = np.einsum('krm->m', self.phi_zeta)
        self.beta_pi = self.xi_0 + beta_temp[::-1].cumsum()[::-1] - beta_temp

    def _update_q_rho(self):
        """
        A method for computing the variational approximation for each \rho_{km}.
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
        
    def _compute_full_ELBO(self):
        """
        """
        pass

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
        self.ELBO_store = np.zeros((n_CAVI_its, num_grad_steps, self.M_w))
        
        self.phi_w_track = np.zeros((n_CAVI_its + 1, self.num_nodes, self.M_w))
        self.phi_w_track[0,:,:] = self.phi_w.copy()
        self.phi_u_track = np.zeros((n_CAVI_its + 1, self.num_layers, 
                                     self.num_nodes, self.M_u))
        self.phi_u_track[0,:,:,:] = self.phi_u.copy()
        self.phi_zeta_track = np.zeros((n_CAVI_its + 1, self.num_nodes, self.M_zeta))
        self.phi_zeta_track[0,:,:] = self.phi_zeta.copy()
        
        for CAVI_rep in range(n_CAVI_its):
            print(f"Iteration {CAVI_rep + 1} of {n_CAVI_its}")

            print("Updating u")
            self._update_q_u()

            print("Updating w")
            self._update_q_w(num_mc)
            self.phi_w_track[CAVI_rep + 1,:,:] = self.phi_w.copy()
            
            print("Updating zeta")
            self._update_q_zeta()

            print("Updating phi")
            self._update_q_phi(num_mc, num_grad_steps, CAVI_rep, alpha, beta1,
                               beta2, eps)
            
            print("Updating phi_0")
            self._update_q_phi0()
            
            print("Updating sigma2")
            self._update_q_sigma2()
            
            print("Updating gamma")
            self._update_q_gamma()
            
            print("Updating pi")
            self._update_q_pi()

            print("Updating rho")
            self._update_q_rho()

            

# %%
