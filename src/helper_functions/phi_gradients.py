import numpy as np

from scipy.stats import norm
from scipy import linalg

def _compute_M_from_sigma(sigma):
    """
    """
    L = linalg.cholesky(sigma, lower=True)
    
    M = np.tril(L, k=-1) 
    np.fill_diagonal(M, np.log(np.diag(L)))
    
    return M

def _compute_L_from_M(M):
    """
    """
    L = np.tril(M, k=-1)
    np.fill_diagonal(L, np.exp(np.diag(M)))
    
    return L

def _compute_sigma_and_inv_from_M(M):
    """
    """
    L = _compute_L_from_M(M)
    
    sigma = L @ L.T
    sigma_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
    
    return sigma, sigma_inv

def compute_gradient_theta_phi_k(theta_phi_k: np.array, k: int, P: int, _sample_phi, 
                                 num_mc: int, features: np.array, sigma_phi: np.array,
                                 sigma_phi_inv: np.array,phi_w: np.array, nu_sigma2: float, 
                                 omega_sigma2: float, theta_phi0: np.array):
    """
    Parameters:
        - theta_phi_k: (P, ) the parameter to optimise.
        - k: index for phi_k.
        - P: features length.
        - _sample_phi: function for sampling phi_k
        - num_mc: number of MC samples.
        - features: (num_nodes, P) the features for all nodes. 
        - sigma_phi: (M_w, P, P).
        - sigma_phi_inv: (M_w, P, P).
        - phi_w: (num_nodes, M_w).
        - nu_sigma2: (M_w, ).
        - omega_sigma2: (M_w, ).
        - theta_phi0: (M_w, P).
    Output:
        - grad_theta_k: (P, ) gradient of ELBO wrt theta_phi.
    """
    grad_theta_k = np.zeros((P))

    # Sample phi and compute delta using current variational parameter values
    phi_k_samps = _sample_phi(num_mc, theta_phi_k, sigma_phi[k],
                              single=True) # (features_length, num_mc)

    # Compute the phi - theta
    vec = (phi_k_samps - np.tile(theta_phi_k, (num_mc, 1))) # (num_mc, features_length)
    
    # Once with einsum
    einsum_term = np.einsum('np,cp->nc', features, phi_k_samps, optimize=True) # (num_nodes, num_mc)
    cdf_term = norm.logcdf(einsum_term)
    cdf_term_sub = norm.logcdf(-einsum_term)

    # Multiply and avergage to get mc approx
    mc_approx = np.einsum('cp,nc->np', vec, cdf_term, optimize=True) / num_mc # (num_nodes, features_length)
    mc_approx_sub = np.einsum('cp,nc->np', vec, cdf_term_sub, optimize=True) / num_mc
    
    # Scale by matrix
    scaled = np.einsum('ij,nj->ni', sigma_phi_inv[k], mc_approx, optimize=True) # (num_nodes, features_length)
    scaled_sub = np.einsum('ij,nj->ni', sigma_phi_inv[k], mc_approx_sub, optimize=True)

    # Sum over phi_w
    grad_theta_k += np.einsum('n,np->p', phi_w[:,k], scaled)
    grad_theta_k += np.einsum('n,np->p',
                                (np.cumsum(phi_w[:,::-1], axis=1)[:, ::-1] - phi_w)[:,k], 
                                scaled_sub)
    # Subtract remaining terms (CHANGED TO SUBTRACT)
    grad_theta_k -= ((nu_sigma2[k] / omega_sigma2[k]) * 
                     (theta_phi_k - theta_phi0[k]))
                
    # We want to maximise the ELBO, so negate for ADAM
    grad_theta_k = -grad_theta_k

    # print(f"grad_theta_{k}: {grad_theta_k}")

    return grad_theta_k

def compute_gradient_M_phi_k(M_phi_k: np.array, k: int, P: int, _sample_phi, 
                             theta_phi: np.array, num_mc: int, features: np.array, 
                             phi_w: np.array, nu_sigma2: float, omega_sigma2: float, 
                             theta_phi0: np.array):
    """
    Parameters:
        - M_phi_k: (P, P) the parameter to optimise.
        - k: index for M_phi_k.
        - P: features length.
    """
    grad_sigma_k = np.zeros((P, P))
    grad_M_k = np.zeros((P, P))
    
    sigma_phi_k, sigma_phi_inv_k = (
        _compute_sigma_and_inv_from_M(M_phi_k)
    )
    
    # Sample phi and compute delta using current variational parameter values
    phi_k_samps = _sample_phi(num_mc, theta_phi[k], sigma_phi_k,
                              single=True) # (features_length, num_mc)

    # Compute the phi - theta
    vec = (phi_k_samps - np.tile(theta_phi[k], (num_mc, 1))) # (num_mc, features_length)
    
    # Once with einsum
    einsum_term = np.einsum('np,cp->nc', features, phi_k_samps, optimize=True) # (num_nodes, num_mc)
    cdf_term = norm.logcdf(einsum_term)  # Shift away from 0
    cdf_term_sub = norm.logcdf(-einsum_term) # Shift away from 0
    
    # Compute outer product and add matrices
    scaled_outer_prod = (
        np.einsum('ia,ma,mb,bj->ijm', 
                sigma_phi_inv_k, vec, vec, sigma_phi_inv_k, 
                optimize=True)
        - sigma_phi_inv_k[:,:,np.newaxis] # (P,P,num_mc)
    )
    
    mc_approx = 0.5 * np.einsum('ijm,nm->nij', 
                                scaled_outer_prod, cdf_term,
                                optimize=True) / num_mc # (num_nodes, P, P)

    mc_approx_sub = 0.5 * np.einsum('ijm,nm->nij',
                                    scaled_outer_prod, cdf_term_sub,
                                    optimize=True) / num_mc
    # Sum over phi_w
    grad_sigma_k += np.einsum('n,nij->ij', 
                                phi_w[:,k], mc_approx,
                                optimize=True)
    
    grad_sigma_k += np.einsum('n,npq->pq',
                                (np.cumsum(phi_w[:,::-1], axis=1)[:, ::-1] - phi_w)[:,k], 
                                mc_approx_sub)        

    # Add on the remaining terms
    grad_sigma_k += 0.5 * (
            -nu_sigma2[k] / omega_sigma2[k] * np.eye(P) +
            sigma_phi_inv_k
        )
    
    # Convert from Sigma_K gradient to M_k gradient
    L_k = _compute_L_from_M(M_phi_k)
    for i in range(P):
        for j in range(i+1):
            if i == j:
                E = np.zeros_like(M_phi_k)
                E[i,i] = 1
                grad_M_k[i,i] = (
                    np.trace(grad_sigma_k.T @ (np.exp(M_phi_k[i,i]) * (E @ L_k.T + L_k @ E.T)))
                )
                # grad_sigma_M_k = np.exp(M_phi_k[i,i]) * (E @ L_k.T + L_k @ E.T)
            else: # CHECK IF i < j?
                E = np.zeros_like(M_phi_k)
                E[i,j] = 1
                grad_M_k[i,j] = (
                    np.trace(grad_sigma_k.T @ (E @ L_k.T + L_k @ E.T))
                )
            #     grad_sigma_M_k = (E @ L_k.T + L_k @ E.T)
            # else:
            #     pass
            # grad_M_k[i,j] = np.trace(grad_sigma_k.T @ grad_sigma_M_k)
                
    # We want to maximise ELBO, so negate for ADAM
    grad_M_k = -grad_M_k
    
    # print(f"grad_M_{k}: {grad_M_k}")
    
    return grad_M_k