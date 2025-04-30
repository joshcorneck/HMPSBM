
"""
A script containing functions for reconstructing parameters that 
are derived.
"""

import numpy as np
from scipy.stats import norm

def sample_phi(num_mc: int, M_w: int, P: int, theta_phi: np.array,
                sigma_phi: np.array):
    """
    A method to sample values for phi for the current estimates of
    the variationl parameters.
    Parameters:
        - num_mc: the number of MC samples.
        - M_w: truncated number of global groups.
        - P: number of features.
        - theta_phi: posterior mean (M_w x P)
        - sigma_phi: posterior variances (M_w x P x P)
    """
    phi_k_samps = np.zeros((M_w, P, num_mc))
    rng = np.random.default_rng() # Speed improvement with this sampler
    for k in range(M_w):
        phi_k_samps[k,:,:] = rng.multivariate_normal(theta_phi[k,:],
                                                        sigma_phi[k,:,:],
                                                        size=num_mc).T
            
    return phi_k_samps

def compute_tau(features: np.array, M_w: int, num_mc: int = None, 
                theta_phi: np.array = None, 
                sample_phi_bool: bool = False):
    """
    Reconstruct values of tau by sampling phi_k num_mc times to get
    a posterior mean estimate. Alternatively, can supply theta_phi
    as a MAP estimate of the phi_k.
    
    Parameters:
        - num_mc: number of samples to approximate phi_k.
        - M_w: truncation of global groups.
        - features: array of features (N x P).
        - theta_phi: array of MAP estimates (M_w x P).
        - sample_phi_bool: Boolean for whether to do MC approximation.
    """
    num_nodes, _ = features.shape

    if sample_phi_bool:
        phi_k_samps = sample_phi(num_mc)
        delta_ik_samps = np.zeros((num_nodes, M_w, num_mc))
        for i in range(num_nodes):
            for k in range(M_w):
                delta_ik_samps[i,k,:] = np.dot(features[i,:], phi_k_samps[k])
        
        tau_ik_samps = np.zeros_like(delta_ik_samps)
        for i in range(num_nodes):
            for k in range(M_w):
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
        
        return tau_ik_samps.mean(axis=2)
        
    else:
        if theta_phi is None:
            raise ValueError("Supply theta_phi.")
        
        delta_ik = np.zeros((num_nodes, M_w))
        for i in range(num_nodes):
            for k in range(M_w):
                delta_ik[i,k] = np.dot(features[i,:], theta_phi[k])
                
        tau_ik = np.zeros_like(delta_ik)
        for i in range(num_nodes):
            for k in range(M_w):
                if k == 0:
                    tau_ik[i,k] = norm.cdf(delta_ik[i,k])
                else:
                    tau_ik[i,k] = (
                        norm.cdf(delta_ik[i,k]) *
                        (1 - norm.cdf(delta_ik[i,:k])).prod()
                    )
            
        # Normalise
        row_sums = tau_ik.sum(axis=1, keepdims=True)

        # Avoid division by zero by setting zero sums to 1 temporarily
        row_sums[row_sums == 0] = 1

        # Perform element-wise division, rows with original zero sums will remain unchanged
        tau_ik = tau_ik / row_sums
        
        return tau_ik

def reconstruct_gamma(alpha: np.array, beta: np.array):
    """
    Reconstruct gamma vectors using MAP estimates.
    
    Parameters:
        - alpha: array of shape (M_u x M_u).
        - beta: array of shape (M_u x M_u)
    """
    gamma = np.zeros_like(alpha)
    for k in range(gamma.shape[0]):
        for s in range(gamma.shape[1]):
            if s == 0:
                gamma[k,s] = alpha[k,s] / (alpha[k,s] + beta[k,s])
            else:
                gamma[k,s] = (
                    alpha[k,s] / (alpha[k,s] + beta[k,s]) 
                )
                gamma[k,s] *= (1 - alpha[k,:s] / (alpha[k,:s] + beta[k,:s])).prod()

    return gamma / gamma.sum(axis=1, keepdims=True)
