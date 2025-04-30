"""
A script containing different implementations of the ADAM
update procedure.
"""

import numpy as np
import torch
from typing import Callable

from scipy.stats import norm
from src.helper_functions.phi_gradients import (
    _compute_L_from_M, _compute_M_from_sigma, _compute_sigma_and_inv_from_M,
    compute_gradient_theta_phi_k, compute_gradient_M_phi_k
)

def ADAM_single(theta_phi: np.array, sigma_phi: np.array, phi_w: np.array,
               M_w: int, P: int, num_mc: int, features: np.array, 
               nu_sigma2: np.array, omega_sigma2: np.array, theta_phi0: np.array,
               first_moment_theta_best: np.array, first_moment_M_best: np.array, 
               second_moment_theta_best: np.array, second_moment_M_best: np.array,
               max_num_grad_steps: int, max_decrease_count: int,
               _compute_full_ELBO: Callable, _compute_phi_ELBO: Callable, 
               _sample_phi: Callable, 
               alpha_theta: float, alpha_sigma: float,
               alpha_set_theta: list, alpha_set_sigma: list, 
               step_theta: np.array, step_sigma: np.array,
               beta1: float, beta2: float, eps: float, multiple_lr: bool):
    """
    Parameters:
        - theta_phi: (M_w, P).
        - sigma_phi: (M_w, P, P).
        - M_w: number of global groups.
        - P: dimension of features.
        - first_moment_theta_best: (M_W, P).
        - first_moment_M_best: (M_w, P, P).
    """
    # Initialise temporary parameters
    theta_phi_best = theta_phi.copy()
    theta_phi_curr = theta_phi.copy()
    sigma_phi_best = sigma_phi.copy()
    sigma_phi_curr = sigma_phi.copy()
    sigma_phi_inv_best = np.zeros((M_w, P, P))
    M_best = np.zeros((M_w, P, P))
    for m in range(M_w): # RECODE - INEFFICIENT.
        M_best[m,:,:] = _compute_M_from_sigma(sigma_phi_best[m,:,:])
        _, sigma_phi_inv_best[m,:,:] = (
            _compute_sigma_and_inv_from_M(M_best[m,:,:])
            )
    sigma_phi_inv_curr = sigma_phi_inv_best.copy()
    M_curr = M_best.copy()
    
    for k in range(M_w):
        ## In the outer loop, we do joint updates for \theta_k and \Sigma_k 
        ## as these are dependent.

        ##########################
        ##~~ ADAM for theta_k ~~##
        ##########################

        # Compute initial ELBO value
        ELBO_best_k = _compute_phi_ELBO(theta_phi=theta_phi_best,
                                        sigma_phi=sigma_phi_best)

        # ADAM parameters
        first_moment_theta_curr = first_moment_theta_best.copy()[k]
        first_moment_M_curr = first_moment_M_best.copy()[k]
        second_moment_theta_curr = second_moment_theta_best.copy()[k]
        second_moment_M_curr = second_moment_M_best.copy()[k]

        decrease_counter = 0
        end_step_reduct = False
        # print(f"...Updating theta_{k}...")
        for step in range(max_num_grad_steps):
            print(f"...theta_{k} iteration {step + 1}...", end='\r')
            ## Do ADAM for the inner loop.
            
            #### 
            #~~ Estimate the expectation for theta_k
            ####
            grad_theta_k = compute_gradient_theta_phi_k(
                theta_phi_k=theta_phi_curr[k], k=k, P=P,
                _sample_phi=_sample_phi, num_mc=num_mc, 
                features=features, sigma_phi=sigma_phi_best,
                sigma_phi_inv=sigma_phi_inv_best, phi_w=phi_w, 
                nu_sigma2=nu_sigma2, omega_sigma2=omega_sigma2,
                theta_phi0=theta_phi0
            )
            
            ####
            #~~ ADAM Steps
            ####
            # CHANGED THE CODE HERE TO ALLOW IT TO TAKE A NUMBER OF STEPS BEFORE REJECTING COMPLETELY
            first_moment_theta_curr = beta1 * first_moment_theta_curr + (1 - beta1) * grad_theta_k
            second_moment_theta_curr = beta2 * second_moment_theta_curr + (1 - beta2) * grad_theta_k ** 2
            first_moment_theta_bias_curr = first_moment_theta_curr / (1 - beta1 ** step_theta[k])
            second_moment_theta_bias_curr = second_moment_theta_curr / (1 - beta2 ** step_theta[k])

            if multiple_lr:
                ## Select optimal value of alpha
                ELBO_eval_k = np.zeros((len(alpha_set_theta)))
                theta_phi_eval = theta_phi_curr.copy()
                for idx_theta, alpha_theta in enumerate(alpha_set_theta):
                    # Updating theta_k
                    theta_phi_eval[k] = (
                                theta_phi_curr[k] 
                                - alpha_theta * first_moment_theta_bias_curr / (np.sqrt(second_moment_theta_bias_curr) + eps)
                            ).copy()  
                    ELBO_eval_k[idx_theta] = (
                            _compute_phi_ELBO(theta_phi=theta_phi_eval, 
                                            sigma_phi=sigma_phi_curr)
                            )
                ## Compute parameter updates
                alpha_theta = alpha_set_theta[np.argmax(ELBO_eval_k)]
                
            theta_phi_curr[k] = (
                        theta_phi_curr[k] 
                        - alpha_theta * first_moment_theta_bias_curr / (np.sqrt(second_moment_theta_bias_curr) + eps)
                    ).copy() 

            ELBO_curr_k = (
                    _compute_phi_ELBO(theta_phi=theta_phi_curr, 
                                      sigma_phi=sigma_phi_best)
                )
            # print(f"ELBO_curr_theta_{k}: {ELBO_curr_k}")
            
            ## Checking ELBO
            if ELBO_curr_k > ELBO_best_k:
                # print("ELBO increased", end='\r')
                ELBO_best_k = ELBO_curr_k # Update the best ELBO
                decrease_counter = 0 # Set decreasing counter to 0
                
                # ELBO increased so update best parameters
                theta_phi_best = theta_phi_curr.copy()
                first_moment_theta_best[k] = first_moment_theta_curr.copy()
                second_moment_theta_best[k] = second_moment_theta_curr.copy()
                
                # Increase steps
                step_theta[k] += 1
            else:
                # print("ELBO decreased", end='\r')
                decrease_counter += 1
                # Increase steps
                step_theta[k] += 1
                if decrease_counter == max_decrease_count:
                    # Shift the number of steps back
                    step_theta[k] -= int(decrease_counter)
                    end_step_reduct = True
                    # print("Maximum number of decreases reached.", end='\r')
                    break
        
        if not end_step_reduct:
            step_theta[k] -= int(decrease_counter)
                
        ##########################
        ##~~ ADAM for Sigma_k ~~##
        ##########################
        
        # Initialise temporary parameters
        # Compute initial ELBO value
        ELBO_best_k = _compute_phi_ELBO(theta_phi=theta_phi_best,
                                        sigma_phi=sigma_phi_best)
        # print(f"Initial ELBO_best_k_sigma: {ELBO_best_k}")
        
        decrease_counter = 0
        end_step_reduct = False
        # print(f"...Updating Sigma_{k}...")
        for step in range(max_num_grad_steps):
            print(f"...Sigma_{k} iteration {step + 1}...", end='\r')
            ## Do ADAM for the inner loop.
            
            #### 
            #~~ Estimate the expectation for Sigma_k
            ####
            grad_M_k = compute_gradient_M_phi_k(
                M_phi_k=M_curr[k], k=k, P=P, _sample_phi=_sample_phi,
                theta_phi=theta_phi_best, num_mc=num_mc, features=features,
                phi_w=phi_w, nu_sigma2=nu_sigma2, omega_sigma2=omega_sigma2,
                theta_phi0=theta_phi0
            )
            
            first_moment_M_curr = beta1 * first_moment_M_curr + (1 - beta1) * grad_M_k
            second_moment_M_curr = beta2 * second_moment_M_curr + (1 - beta2) * grad_M_k ** 2                
            first_moment_M_bias_curr = first_moment_M_curr / (1 - beta1 ** step_sigma[k])
            second_moment_M_bias_curr = second_moment_M_curr / (1 - beta2 ** step_sigma[k])
            
            ## Select optimal value of alpha
            if multiple_lr:
                ELBO_eval_k = np.zeros((len(alpha_set_sigma)))
                M_eval = M_curr.copy()
                sigma_phi_eval = sigma_phi_curr.copy()
                for idx_sigma, alpha_sigma in enumerate(alpha_set_sigma):
                    # Updating Sigma_k
                    M_eval[k] = (
                        M_curr[k] 
                        - alpha_sigma * first_moment_M_bias_curr / (np.sqrt(second_moment_M_bias_curr) + eps)
                        ).copy()
                    # Convert back to sigma_phi 
                    sigma_phi_eval[k], _ = (
                        _compute_sigma_and_inv_from_M(M_eval[k])
                        )
                    ELBO_eval_k[idx_sigma] = (
                            _compute_phi_ELBO(theta_phi=theta_phi_best, 
                                            sigma_phi=sigma_phi_eval)
                            )
                ## Compute parameter updates
                alpha_sigma = alpha_set_sigma[np.argmax(ELBO_eval_k)]
                
            # Updating Sigma_k
            M_curr[k] = (
                M_curr[k] 
                - alpha_sigma * first_moment_M_bias_curr / (np.sqrt(second_moment_M_bias_curr) + eps)
                ).copy()
            # Convert back to sigma_phi 
            sigma_phi_curr[k], sigma_phi_inv_curr[k] = (
                _compute_sigma_and_inv_from_M(M_curr[k])
                )
            ELBO_curr_k = (
                    _compute_phi_ELBO(theta_phi=theta_phi_best, 
                                      sigma_phi=sigma_phi_curr)
                )
            # print(f"ELBO_curr_Sigma_{k}: {ELBO_curr_k}")
                
            ## Checking ELBO
            if ELBO_curr_k > ELBO_best_k:
                # print("ELBO increased", end='\r')
                ELBO_best_k = ELBO_curr_k # Update the best ELBO
                decrease_counter = 0 # Set decreasing counter to 0
                
                # ELBO increased so update best parameters
                sigma_phi_best = sigma_phi_curr.copy()
                sigma_phi_inv_best = sigma_phi_inv_curr.copy()
                M_best = M_curr.copy()
                first_moment_M_best[k] = first_moment_M_curr.copy()
                second_moment_M_best[k] = second_moment_M_curr.copy()
                
                # Increase steps
                step_sigma[k] += 1
            else:
                print("ELBO decreased", end='\r')
                decrease_counter += 1
                # Increase steps
                step_sigma[k] += 1
                if decrease_counter == max_decrease_count:
                    # Shift the number of steps back
                    step_sigma[k] -= int(decrease_counter)
                    end_step_reduct = True
                    # print("Maximum number of decreases reached.", end='\r')
                    break
                
    if not end_step_reduct:
        step_sigma[k] -= int(decrease_counter)
                         
    return (
        theta_phi_best, sigma_phi_best,
        step_theta, step_sigma,
        first_moment_theta_best, first_moment_M_best,
        second_moment_theta_best, second_moment_M_best
        )


def ADAM_single_torch(theta_phi: np.array, sigma_phi: np.array, M_w: int,
                      P:int, max_decrease_count: int, max_num_grad_steps: int,
                      _compute_phi_ELBO: Callable, _sample_phi: Callable,
                      num_mc: int, features: np.array, phi_w: np.array,
                      nu_sigma2: np.array, omega_sigma2: np.array,
                      theta_phi0: np.array, lr_theta: float,
                      lr_sigma: float):
    """
    """
    # Copy all relevant numpy arrays
    theta_phi_curr = theta_phi.copy()
    theta_phi_best = theta_phi.copy()
    
    sigma_phi_curr = sigma_phi.copy()
    sigma_phi_best = sigma_phi.copy()
    sigma_phi_inv_best = np.zeros((M_w, P, P))
    M_phi_best = np.zeros((M_w, P, P))
    for m in range(M_w): 
        M_phi_best[m,:,:] = _compute_M_from_sigma(sigma_phi_best[m,:,:])
        _, sigma_phi_inv_best[m,:,:] = (
            _compute_sigma_and_inv_from_M(M_phi_best[m,:,:])
            )
    sigma_phi_inv_curr = sigma_phi_inv_best.copy()
    M_phi_curr = M_phi_best.copy()
    for k in range(M_w):
        ### ADAM for theta_k ###
        
        # Instantiate torch tensor
        theta_phi_k = torch.tensor(theta_phi[k], dtype=torch.float32)
        optimizer = torch.optim.Adam([theta_phi_k], lr=lr_theta) # LEARNING RATE
    
        
        # Evaluate ELBO
        ELBO_theta_k = _compute_phi_ELBO(theta_phi=theta_phi_curr,
                                         sigma_phi=sigma_phi_best)
        
        decrease_counter = 0
        for update in range(max_num_grad_steps):
            # Clear gradient
            optimizer.zero_grad()
            
            if update == 0:
                ELBO_theta_k_best = ELBO_theta_k
            
            # Compute and apply gradient
            grad_theta_k = compute_gradient_theta_phi_k(
                theta_phi_k=theta_phi_curr[k], k=k, P=P,
                _sample_phi=_sample_phi, num_mc=num_mc, 
                features=features, sigma_phi=sigma_phi_best,
                sigma_phi_inv=sigma_phi_inv_best, phi_w=phi_w, 
                nu_sigma2=nu_sigma2, omega_sigma2=omega_sigma2,
                theta_phi0=theta_phi0
            )            
            theta_phi_k.grad = torch.tensor(grad_theta_k, dtype=torch.float32)
            optimizer.step()
            
            # Update current value
            theta_phi_k_curr = theta_phi_k.detach().numpy()
            theta_phi_curr[k] = theta_phi_k_curr.copy()
            
            # Evaluate ELBO
            ELBO_theta_k = _compute_phi_ELBO(theta_phi=theta_phi_curr,
                                             sigma_phi=sigma_phi_best)
            
            # Check if ELBO improved
            if ELBO_theta_k > ELBO_theta_k_best:
                ELBO_theta_k_best = ELBO_theta_k
                theta_phi_best = theta_phi_curr.copy()
                decrease_counter = 0  # Reset counter on improvement
            else:
                decrease_counter += 1
                # If too many decreases, revert and break
                if decrease_counter >= max_decrease_count:
                    print(f"Reverting to best parameters after {max_decrease_count} decreases")
                    break
            
            # # Print progress
            # print(f"ELBO: {ELBO_theta_k}")
            
        print(f"...Finished ADAM for theta_{k}...")
            
        ### ADAM for Sigma_k ###
        
        # Instantiate torch tensor
        M_phi_k = torch.tensor(M_phi_best[k], dtype=torch.float32)
        optimizer = torch.optim.Adam([M_phi_k], lr=lr_sigma) # LEARNING RATE
        
        # Evaluate ELBO
        ELBO_Sigma_k = _compute_phi_ELBO(theta_phi=theta_phi_best,
                                         sigma_phi=sigma_phi_curr)
        
        decrease_counter = 0
        for update in range(max_num_grad_steps):
            # Clear gradient
            optimizer.zero_grad()
            
            if update == 0:
                ELBO_Sigma_k_best = ELBO_Sigma_k
            
            # Compute and apply gradient
            grad_M_k = compute_gradient_M_phi_k(
                M_phi_k=M_phi_curr[k], k=k, P=P, _sample_phi=_sample_phi,
                theta_phi=theta_phi_best, num_mc=num_mc, features=features,
                phi_w=phi_w, nu_sigma2=nu_sigma2, omega_sigma2=omega_sigma2,
                theta_phi0=theta_phi0
            )         
            M_phi_k.grad = torch.tensor(grad_M_k, dtype=torch.float32)
            optimizer.step()
            
            # Update current value
            M_phi_curr[k] = M_phi_k.detach().numpy().copy()
            sigma_phi_curr[k], sigma_phi_inv_curr[k] = (
                _compute_sigma_and_inv_from_M(M_phi_curr[k])
                )
            # Evaluate ELBO
            ELBO_Sigma_k = _compute_phi_ELBO(theta_phi=theta_phi_best,
                                             sigma_phi=sigma_phi_curr)
            
            # Check if ELBO improved
            if ELBO_Sigma_k > ELBO_Sigma_k_best:
                ELBO_Sigma_k_best = ELBO_Sigma_k
                M_phi_best = M_phi_curr.copy()
                sigma_phi_best = sigma_phi_curr.copy()
                sigma_phi_inv_best = sigma_phi_inv_curr.copy()
                decrease_counter = 0  # Reset counter on improvement
            else:
                decrease_counter += 1
                # If too many decreases, revert and break
                if decrease_counter >= max_decrease_count:
                    print(f"Reverting to best parameters after {max_decrease_count} decreases")
                    break
            
            # # Print progress
            # print(f"ELBO: {ELBO_Sigma_k}")
            
        print(f"...Finished ADAM for Sigma_{k}...")

    return (theta_phi_best, sigma_phi_best)

def ADAM_joint(theta_phi: np.array, sigma_phi: np.array, phi_w: np.array,
                    M_w: int, P: int, num_mc: int, features: np.array, 
                    nu_sigma2: np.array, omega_sigma2: np.array, theta_phi0: np.array,
                    first_moment_theta_best: np.array, first_moment_M_best: np.array, 
                    second_moment_theta_best: np.array, second_moment_M_best: np.array,
                    max_num_grad_steps: int, max_decrease_count: int,
                    _compute_full_ELBO: Callable, _compute_phi_ELBO: Callable, 
                    _sample_phi: Callable, 
                    alpha_theta: float, alpha_sigma: float,
                    alpha_set_theta: list, alpha_set_sigma: list, 
                    step_theta: np.array, step_sigma: np.array,
                    beta1: float, beta2: float, eps: float, multiple_lr: bool):
    """
    """
    for k in range(M_w):
        ## In the outer loop, we do joint updates for \theta_k and \Sigma_k 
        ## as these are dependent.

        # Initialise temporary parameters
        theta_phi_best = theta_phi.copy()
        theta_phi_curr = theta_phi.copy()
        sigma_phi_best = sigma_phi.copy()
        sigma_phi_curr = sigma_phi.copy()
        sigma_phi_inv_best = np.zeros((M_w, P, P))
        M_best = np.zeros((M_w, P, P))
        for m in range(M_w): # RECODE - INEFFICIENT.
            M_best[m,:,:] = _compute_M_from_sigma(sigma_phi_best[m,:,:])
            _, sigma_phi_inv_best[m,:,:] = (
                _compute_sigma_and_inv_from_M(M_best[m,:,:])
                )
        sigma_phi_inv_curr = sigma_phi_inv_best.copy()
        M_curr = M_best.copy()

        # Compute initial ELBO value
        ELBO_best_k = _compute_phi_ELBO(theta_phi=theta_phi_best,
                                        sigma_phi=sigma_phi_best)

        # ADAM parameters
        first_moment_theta_curr = first_moment_theta_best.copy()[k]
        first_moment_M_curr = first_moment_M_best.copy()[k]
        second_moment_theta_curr = second_moment_theta_best.copy()[k]
        second_moment_M_curr = second_moment_M_best.copy()[k]

        decrease_counter = 0
        end_step_reduct = False
        for step in range(max_num_grad_steps):
            print(f"...theta_{k} and sigma_{k} iteration {step + 1}...", end='\r')
            ## Do ADAM for the inner loop.
            
            #### 
            #~~ Estimate the expectation for theta_k
            ####
            grad_theta_k = compute_gradient_theta_phi_k(
                theta_phi_k=theta_phi_curr[k], k=k, P=P,
                _sample_phi=_sample_phi, num_mc=num_mc, 
                features=features, sigma_phi=sigma_phi_curr,
                sigma_phi_inv=sigma_phi_inv_curr, phi_w=phi_w, 
                nu_sigma2=nu_sigma2, omega_sigma2=omega_sigma2,
                theta_phi0=theta_phi0
            )
            
            #### 
            #~~ Estimate the expectation for Sigma_k
            ####
            grad_M_k = compute_gradient_M_phi_k(
                M_phi_k=M_curr[k], k=k, P=P, _sample_phi=_sample_phi,
                theta_phi=theta_phi_curr, num_mc=num_mc, features=features,
                phi_w=phi_w, nu_sigma2=nu_sigma2, omega_sigma2=omega_sigma2,
                theta_phi0=theta_phi0
            )
            
            ####
            #~~ ADAM Steps
            ####
            # CHANGED THE CODE HERE TO ALLOW IT TO TAKE A NUMBER OF STEPS BEFORE REJECTING COMPLETELY
            first_moment_theta_curr = beta1 * first_moment_theta_curr + (1 - beta1) * grad_theta_k
            second_moment_theta_curr = beta2 * second_moment_theta_curr + (1 - beta2) * grad_theta_k ** 2
            first_moment_M_curr = beta1 * first_moment_M_curr + (1 - beta1) * grad_M_k
            second_moment_M_curr = beta2 * second_moment_M_curr + (1 - beta2) * grad_M_k ** 2                

            first_moment_theta_bias_curr = first_moment_theta_curr / (1 - beta1 ** step_theta[k])
            second_moment_theta_bias_curr = second_moment_theta_curr / (1 - beta2 ** step_theta[k])
            first_moment_M_bias_curr = first_moment_M_curr / (1 - beta1 ** step_sigma[k])
            second_moment_M_bias_curr = second_moment_M_curr / (1 - beta2 ** step_sigma[k])
            
            ## Select optimal value of alpha
            if multiple_lr:
                ELBO_eval_k = np.zeros((len(alpha_set_theta), len(alpha_set_sigma)))
                theta_phi_eval = theta_phi_curr.copy()
                M_eval = M_curr.copy()
                sigma_phi_eval = sigma_phi_curr.copy()
                for idx_theta, alpha_theta in enumerate(alpha_set_theta):
                    for idx_sigma, alpha_sigma in enumerate(alpha_set_sigma):
                        # Updating theta_k
                        theta_phi_eval[k] = (
                                    theta_phi_curr[k] 
                                    - alpha_theta * first_moment_theta_bias_curr / (np.sqrt(second_moment_theta_bias_curr) + eps)
                                ).copy()  
                        # Updating Sigma_k
                        M_eval[k] = (
                            M_curr[k] 
                            - alpha_sigma * first_moment_M_bias_curr / (np.sqrt(second_moment_M_bias_curr) + eps)
                            ).copy()
                        # Convert back to sigma_phi 
                        sigma_phi_eval[k], _ = (
                            _compute_sigma_and_inv_from_M(M_eval[k])
                            )
                        ELBO_eval_k[idx_theta, idx_sigma] = (
                                _compute_phi_ELBO(theta_phi=theta_phi_eval, 
                                                sigma_phi=sigma_phi_eval)
                                )
                        
                ## Compute parameter updates
                best_idx = np.unravel_index(np.argmax(ELBO_eval_k), ELBO_eval_k.shape)
                alpha_theta = alpha_set_theta[best_idx[0]]
                alpha_sigma = alpha_set_sigma[best_idx[1]]
            theta_phi_curr[k] = (
                        theta_phi_curr[k] 
                        - alpha_theta * first_moment_theta_bias_curr / (np.sqrt(second_moment_theta_bias_curr) + eps)
                    ).copy()  
            # Updating Sigma_k
            M_curr[k] = (
                M_curr[k] 
                - alpha_sigma * first_moment_M_bias_curr / (np.sqrt(second_moment_M_bias_curr) + eps)
                ).copy()
            # Convert back to sigma_phi 
            sigma_phi_curr[k], sigma_phi_inv_curr[k] = (
                _compute_sigma_and_inv_from_M(M_curr[k])
                )
            ELBO_curr_k = (
                    _compute_phi_ELBO(theta_phi=theta_phi_curr,
                                      sigma_phi=sigma_phi_curr)
                )
            
            # print(f"ELBO_curr_{k}: {ELBO_curr_k}")
            
            ## Checking ELBO
            if ELBO_curr_k > ELBO_best_k:
                print("ELBO increased", end='\r')
                ELBO_best_k = ELBO_curr_k # Update the best ELBO
                decrease_counter = 0 # Set decreasing counter to 0
                
                # ELBO increased so update best parameters
                theta_phi_best = theta_phi_curr.copy()
                M_best = M_curr.copy()
                sigma_phi_best = sigma_phi_curr.copy()
                sigma_phi_inv_best = sigma_phi_inv_curr.copy()
                first_moment_theta_best[k] = first_moment_theta_curr.copy()
                first_moment_M_best[k] = first_moment_M_curr.copy()
                second_moment_theta_best[k] = second_moment_theta_curr.copy()
                second_moment_M_best[k] = second_moment_M_curr.copy()
                
                # Increase steps
                step_theta[k] += 1
                step_sigma[k] += 1
            else:
                print("ELBO decreased", end='\r')
                decrease_counter += 1
                # Increase steps
                step_theta[k] += 1
                step_sigma[k] += 1
                if decrease_counter == max_decrease_count:
                    # Shift the number of steps back
                    step_theta[k] -= int(decrease_counter)
                    step_sigma[k] -= int(decrease_counter)
                    end_step_reduct = True
                    print("Maximum number of decreases reached.", end='\r')
                    break
    
    if not end_step_reduct:
        step_theta[k] -= int(decrease_counter)
        step_sigma[k] -= int(decrease_counter)
    
    return (
        theta_phi_best, sigma_phi_best,
        step_theta, step_sigma,
        first_moment_theta_best, first_moment_M_best,
        second_moment_theta_best, second_moment_M_best
        )