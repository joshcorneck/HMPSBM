"""
A script containing all functions necessary to compute the ELBO.
"""

import numpy as np
from scipy.special import loggamma, digamma, comb


## Functions for computing the log joint
def precompute_values(alpha_rho: np.array, beta_rho: np.array,
                      alpha_gamma: np.array, beta_gamma: np.array):
    """
    """
    # Precompute values to avoid repeated computation
    digamma_alpha_rho = np.zeros_like(alpha_rho)
    digamma_beta_rho = np.zeros_like(beta_rho)
    valid_mask = (alpha_rho  > 0) & (beta_rho > 0)
    digamma_alpha_rho[valid_mask] = (
        digamma(alpha_rho[valid_mask]) 
        - digamma(alpha_rho[valid_mask] + beta_rho[valid_mask])
    )
    digamma_beta_rho[valid_mask] = (
        digamma(beta_rho[valid_mask]) 
        - digamma(alpha_rho[valid_mask] + beta_rho[valid_mask])
    )
    
    digamma_alpha_gamma = (
        digamma(alpha_gamma) - digamma(alpha_gamma + beta_gamma)
        )
    digamma_beta_gamma = (
        digamma(beta_gamma) - digamma(alpha_gamma + beta_gamma)
        )
    
    return (digamma_alpha_rho, digamma_beta_rho, 
            digamma_alpha_gamma, digamma_beta_gamma)

def log_joint_term1(num_layers: int, num_nodes: int, adj_tensor: np.array,
                    phi_u: np.array, num_adj_samps: int,
                    digamma_alpha_rho: np.array,
                    digamma_beta_rho: np.array):
    """
    Term 1: p(A | z, \rho)
    """
    adj_tensor_mask = adj_tensor.copy()
    adj_tensor_sub_mask = (num_adj_samps - adj_tensor).copy()
    for l in range(num_layers):
        np.fill_diagonal(adj_tensor_mask[l,:,:], 0)
        np.fill_diagonal(adj_tensor_sub_mask[l,:,:], 0)
    
    # Precompute matrix of M choose A_{\ell ij}
    log_M_choose = np.zeros_like(adj_tensor, dtype=float)
    for l in range(num_layers):
        for i in range(num_nodes):
            for j in range(num_nodes):
                log_M_choose[l,i,j] = np.log(comb(num_adj_samps, adj_tensor[l,i,j], exact=False))
    
    ELBO_temp = np.einsum('lik,ljm,lij,km->', phi_u, phi_u, 
                    adj_tensor_mask, digamma_alpha_rho,
                    optimize=True)
    
    ELBO_temp += np.einsum('lik,ljm,lij,km->', phi_u, phi_u, 
                    adj_tensor_sub_mask, digamma_beta_rho,
                    optimize=True)
    
    ELBO_temp += np.einsum('lik,ljm,lij->', phi_u, phi_u, log_M_choose,
                    optimize=True)
    
    return ELBO_temp

def log_joint_term2(alpha_0: float, beta_0: float,
                    digamma_alpha_rho: np.array,
                    digamma_beta_rho: np.array):
    """
    Term 2: p(\rho)
    """
    ELBO_temp = (alpha_0 - 1) * digamma_alpha_rho.sum() 
    ELBO_temp += (beta_0 - 1) * digamma_beta_rho.sum()
    
    return ELBO_temp

def log_joint_term3(phi_u: np.array, phi_w: np.array,
                    digamma_alpha_gamma: np.array,
                    digamma_beta_gamma: np.array):
    """
    Term 3: p(z | w, \gamma)
    """
    cumsum_gamma = (
        digamma_alpha_gamma + 
        np.cumsum(digamma_beta_gamma, axis=1) -
        digamma_beta_gamma
        )
        
    ELBO_temp = np.einsum('lik,iw,wk->', phi_u, phi_w,
                        cumsum_gamma,
                        optimize=True)
    
    return ELBO_temp

def log_joint_term4(_compute_log_tau, num_mc: int, 
                    theta_phi: np.array, sigma_phi: np.array, phi_w: np.array):
    """
    Term 4: p(w | \tau)
    """
    # MC step for the expectation
    log_tau_ik_samps = _compute_log_tau(num_mc, theta_phi=theta_phi, sigma_phi=sigma_phi)
    log_tau_expec = log_tau_ik_samps.mean(axis=2)
    
    ELBO_temp = np.einsum('iw,iw->', phi_w, log_tau_expec)
    
    return ELBO_temp

def log_joint_term5(M_w: int, P: int, omega_sigma2: np.array,
                    nu_sigma2: np.array, theta_phi: np.array,
                    theta_phi0: np.array, sigma_phi: np.array,
                    sigma_phi0: np.array):
    """
    Term 5: p(\phi | \phi^0, \sigma^2)
    """
    ELBO_temp = 0
    for k in range(M_w):
        ELBO_temp += (
            -0.5 * P * (np.log(omega_sigma2[k]) - digamma(nu_sigma2[k])) -
            0.5 * nu_sigma2[k] / omega_sigma2[k] * (
                np.dot(theta_phi[k] - theta_phi0[k],
                        theta_phi[k] - theta_phi0[k]) +
                np.trace(sigma_phi[k]) + np.trace(sigma_phi0[k])
            )
        )
        
    return ELBO_temp

def log_joint_term6(M_w: int, theta_phi0: np.array,
                    sigma_phi0: np.array, mu: np.array):
    """
    Term 6: p(\phi^0)
    """
    ELBO_temp = 0
    for k in range(M_w):
        ELBO_temp += (
            -0.5 * (np.dot(theta_phi0[k] - mu,
                           theta_phi0[k] - mu) +
                    np.trace(sigma_phi0[k]))
        )
        
    return ELBO_temp

def log_joint_term7(M_w: int, nu_sigma2: np.array,
                    omega_sigma2: np.array, nu_0: int,
                    omega_0: int):
    """
    Term 7: p(\sigma^2)
    """
    ELBO_temp = 0
    for k in range(M_w):
        ELBO_temp += (
            -(nu_0 + 1) * (np.log(omega_sigma2[k]) - digamma(nu_sigma2[k])) -
            omega_0 * nu_sigma2[k] / omega_sigma2[k]
        )
    
    return ELBO_temp

def log_joint_term8(nu_0: int, digamma_beta_gamma: np.array):
    """
    Term 8: p(\gamma')
    """
    ELBO_temp = (nu_0 - 1) * digamma_beta_gamma.sum()
    
    return ELBO_temp

def expec_log_q_term1(phi_u: np.array):
    """
    Term 1: q(z)
    """
    ELBO_temp = (phi_u * np.log(phi_u + 10e-10)).sum()
    
    return ELBO_temp

def expec_log_q_term2(alpha_rho: np.array, beta_rho: np.array,
                      digamma_alpha_rho: np.array,
                      digamma_beta_rho: np.array):
    """
    Term 2: q(\rho)
    """
    ELBO_temp = (
                (alpha_rho - 1) * digamma_alpha_rho +
                (beta_rho - 1) * digamma_beta_rho +
                loggamma(alpha_rho + beta_rho) -
                loggamma(alpha_rho) - loggamma(beta_rho)
            ).sum()
    
    return ELBO_temp

def expec_log_q_term3(phi_w: np.array):
    """
    Term 3: q(w)
    """
    ELBO_temp = (phi_w * np.log(phi_w + 10e-10)).sum()
    
    return ELBO_temp

def expec_log_q_term4(alpha_gamma: np.array,
                      beta_gamma: np.array,
                      digamma_alpha_gamma: np.array,
                      digamma_beta_gamma: np.array):
    """
    Term 4: q(gamma')
    """
    # Term 4: q(gamma')
    ELBO_temp = (
        (alpha_gamma - 1) * digamma_alpha_gamma + 
        (beta_gamma - 1) * digamma_beta_gamma +
        loggamma(alpha_gamma + beta_gamma) -
        loggamma(alpha_gamma) - loggamma(beta_gamma)
    ).sum()
    
    return ELBO_temp

def expec_log_q_term5(M_w: int, sigma_phi: np.array):
    """
    Term 5: q(\phi)
    """
    ELBO_temp = 0
    for k in range(M_w):
        ELBO_temp += -0.5 * np.log(np.linalg.det(sigma_phi[k]) + 10e-10)
        
    return ELBO_temp

def expec_log_q_term6(M_w: int, sigma_phi0: np.array):
    """
    Term 6: q(\phi^0)
    """
    ELBO_temp = 0
    for k in range(M_w):
        ELBO_temp += -0.5 * np.log(np.linalg.det(sigma_phi0[k]) + 10e-10)

    return ELBO_temp

def expec_log_q_term7(nu_sigma2: np.array, omega_sigma2: np.array):
    """
    Term 7: q(\sigma^2)
    """
    ELBO_temp = (
        (nu_sigma2 + 1) * digamma(nu_sigma2 + 10e-10) 
        - np.log(omega_sigma2 + 10e-10)
        - loggamma(nu_sigma2)- nu_sigma2 
    ).sum()
    
    return ELBO_temp

def compute_full_ELBO(ELBO_parameter: str, _compute_log_tau, num_layers: int, 
                      num_nodes: int, adj_tensor: np.array,
                      num_mc: int, M_u: int, M_w: int, P: int, 
                      num_adj_samps: int,
                      alpha_rho: np.array, beta_rho: np.array,
                      alpha_0: float, beta_0: float,
                      alpha_gamma: np.array, beta_gamma: np.array,
                      phi_u: np.array, phi_w: np.array, 
                      nu_sigma2: np.array, omega_sigma2: np.array,
                      nu_0: float, omega_0: float,
                      theta_phi: np.array, sigma_phi: np.array,
                      mu: np.array,
                      theta_phi0: np.array, sigma_phi0: np.array):
        """
        A function to compute the full ELBO or the ELBO with respect to a
        particular parameter. The parameter can be passed in as an argument.
        """
        if ELBO_parameter in {"z", "w", "gamma", "rho", "full"}:
            # Precompute values 
            (digamma_alpha_rho, 
                digamma_beta_rho, 
                digamma_alpha_gamma, 
                digamma_beta_gamma
                ) = precompute_values(
                    alpha_rho, beta_rho, 
                    alpha_gamma, beta_gamma)
            
        def compute_expected_log_joint():
            """
            """
            expected_log_joint = 0
            
            if ELBO_parameter in {"z", "rho", "full"}:            
                # Term 1: p(A | z, \rho)
                expected_log_joint += log_joint_term1(
                    num_layers, num_nodes, adj_tensor, phi_u, 
                    num_adj_samps, digamma_alpha_rho, digamma_beta_rho
                )
            
            if ELBO_parameter in {"rho", "full"}: 
                # Term 2: p(\rho)
                expected_log_joint += log_joint_term2(
                    alpha_0, beta_0, 
                    digamma_alpha_rho, digamma_beta_rho
                )
            
            if ELBO_parameter in {"z", "w", "gamma", "full"}: 
                # Term 3: p(z | w, \gamma)
                expected_log_joint += log_joint_term3(
                    phi_u, phi_w, 
                    digamma_alpha_gamma, digamma_beta_gamma
                )
            if ELBO_parameter in {"phi", "w", "full"}: 
                # Term 4: p(w | \tau)
                expected_log_joint += log_joint_term4(
                    _compute_log_tau, num_mc, theta_phi, sigma_phi, phi_w
                )
            
            if ELBO_parameter in {"phi", "phi0", "sigma2", "full"}: 
                # Term 5: p(\phi | \phi^0, \sigma^2)
                expected_log_joint += log_joint_term5(
                    M_w, P, omega_sigma2, nu_sigma2,
                    theta_phi, theta_phi0, 
                    sigma_phi, sigma_phi0
                )
                
            if ELBO_parameter in {"phi0", "full"}: 
                # Term 6: p(\phi^0)
                expected_log_joint += log_joint_term6(
                    M_w, theta_phi0, sigma_phi0, mu
                )
            
            if ELBO_parameter in {"sigma2", "full"}:     
                # Term 7: p(\sigma^2)
                expected_log_joint += log_joint_term7(
                    M_w, nu_sigma2, omega_sigma2, nu_0, omega_0
                )
            
            if ELBO_parameter in {"gamma", "full"}: 
                # Term 8: p(\gamma')
                expected_log_joint += log_joint_term8(nu_0, digamma_beta_gamma)
            
            return expected_log_joint
        
        def compute_expected_log_q():
            """
            """
            expected_log_q = 0
            
            if ELBO_parameter in {"z", "full"}: 
                # Term 1: q(z)
                expected_log_q += expec_log_q_term1(phi_u)
            
            if ELBO_parameter in {"rho", "full"}: 
                # Term 2: q(\rho)
                expected_log_q += expec_log_q_term2(alpha_rho, beta_rho,
                                                    digamma_alpha_rho,
                                                    digamma_beta_rho)
            
            if ELBO_parameter in {"w", "full"}: 
                # Term 3: q(w)
                expected_log_q += expec_log_q_term3(phi_w)
            
            if ELBO_parameter in {"gamma", "full"}:     
                # Term 4: q(gamma')
                expected_log_q += expec_log_q_term4(alpha_gamma, beta_gamma,
                                            digamma_alpha_gamma,
                                            digamma_beta_gamma)
            
            if ELBO_parameter in {"phi", "full"}: 
                # Term 5: q(\phi)
                expected_log_q += expec_log_q_term5(M_w, sigma_phi)
            
            if ELBO_parameter in {"phi0", "full"}: 
                # Term 6: q(\phi^0)
                expected_log_q += expec_log_q_term6(M_w, sigma_phi0)
            
            if ELBO_parameter in {"sigma2", "full"}:     
                # Term 7: q(\sigma^2)
                expected_log_q += expec_log_q_term7(nu_sigma2, omega_sigma2)
            
            return expected_log_q
        
        # Compute and store
        ELBO = (
            compute_expected_log_joint() - compute_expected_log_q()
        )
            
        return ELBO