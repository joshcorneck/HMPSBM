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
                print("ELBO increased", end='\r')
                ELBO_best_k = ELBO_curr_k # Update the best ELBO
                decrease_counter = 0 # Set decreasing counter to 0
                
                # ELBO increased so update best parameters
                theta_phi_best = theta_phi_curr.copy()
                first_moment_theta_best[k] = first_moment_theta_curr.copy()
                second_moment_theta_best[k] = second_moment_theta_curr.copy()
                
                # Increase steps
                step_theta[k] += 1
            else:
                print("ELBO decreased", end='\r')
                decrease_counter += 1
                # Increase steps
                step_theta[k] += 1
                if decrease_counter == max_decrease_count:
                    # Shift the number of steps back
                    step_theta[k] -= int(decrease_counter)
                    end_step_reduct = True
                    print("Maximum number of decreases reached.", end='\r')
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
        
        decrease_counter = 0
        end_step_reduct = False
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
                print("ELBO increased", end='\r')
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
                    print("Maximum number of decreases reached.", end='\r')
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

def ADAM_joint_torch():
    """
    """
    pass


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def ADAM_sequential(theta_phi: np.array, sigma_phi: np.array, phi_w: np.array,
#                     M_w: int, P: int, num_mc: int, features: np.array, 
#                     nu_sigma2: np.array, omega_sigma2: np.array, theta_phi0: np.array,
#                     first_moment_theta_best: np.array, first_moment_M_best: np.array, 
#                     max_num_grad_steps: int, max_decrease_count: int,
#                     _compute_full_ELBO: Callable, _compute_phi_ELBO: Callable, 
#                     _sample_phi: Callable, 
#                     alpha_set_theta: list, alpha_set_sigma: list, 
#                     step_theta: np.array, step_sigma: np.array,
#                     beta1: float, beta2: float, eps: float):
#     """
#     """
#     # Boolean for whether parameter has converged
#     param_converged = np.zeros((int(2 * M_w)), dtype=bool)
#     ELBO_phi_dec_track = np.zeros((int(2 * M_w)))
    
#     # Initialise a temporary theta_phi
#     theta_phi_best = theta_phi.copy()
#     theta_phi_curr = theta_phi.copy()
#     sigma_phi_best = sigma_phi.copy()
#     sigma_phi_curr = sigma_phi.copy()
#     sigma_phi_inv_best = np.zeros_like(sigma_phi)
#     sigma_phi_inv_curr = np.zeros_like(sigma_phi)

#     # Compute value of the ELBO
#     ELBO_best = np.zeros((int(2 * M_w), ), dtype=float) # 0:(M_w-1) for theta, M_w:(2*M_w) for Sigma
#     ELBO_best[:] = _compute_full_ELBO()

#     # ADAM parameters
#     first_moment_theta_curr = np.zeros((M_w, P))
#     first_moment_M_curr = np.zeros((M_w, P, P))
#     second_moment_theta_curr = np.zeros((M_w, P))
#     second_moment_M_curr = np.zeros((M_w, P, P))
    
#     first_moment_theta_bias_curr = np.zeros((M_w, P))
#     second_moment_theta_bias_curr = np.zeros((M_w, P))
#     first_moment_M_bias_curr = np.zeros((M_w, P, P))
#     second_moment_M_bias_curr = np.zeros((M_w, P, P))
    
#     # Compute initial values of the M_m
#     M_best = np.zeros((M_w, P, P))
#     for m in range(M_w):
#         M_best[m,:,:] = _compute_M_from_sigma(sigma_phi_best[m,:,:])
#     M_curr = M_best.copy()

#     for step in range(max_num_grad_steps):
#         if np.all(param_converged == True):
#             break
        
#         print(f"Gradient step {step + 1} of {max_num_grad_steps}.", end='\r')

#         # Empty arrays for gradients
#         grad_theta = np.zeros((M_w, P))
#         grad_sigma = np.zeros((M_w, P, P))
#         grad_M = np.zeros((M_w, P, P))
        
#         if step == 1:
#             for m in range(self.M_w):
#                 _, sigma_phi_inv_curr[m,:,:] = (
#                     _compute_sigma_and_inv_from_M(M_curr[m,:,:])
#                 )
#             sigma_phi_inv_best = sigma_phi_inv_curr.copy()
        
#         # Sample phi and compute delta using current variational parameter values
#         phi_k_samps = self._sample_phi(num_mc, theta_phi_best, sigma_phi_best)
        
#         ## Computing normalised versions of \log\Phi(x_i^T\phi_k)
#         # tau_ik_samps, norm_ik_samps = self._compute_tau(num_mc, phi_k_samps, True)
                        
#         ## Estimate the expectation for theta
#         # Compute the phi - theta
#         vec = (phi_k_samps - np.tile(theta_phi_best[:, :, np.newaxis], (1, 1, num_mc))) # (M_w, P, num_mc)
        
#         # Once with einsum
#         einsum_term = np.einsum('np,mpc->nmc', self.features, phi_k_samps, optimize=True) # (num_nodes, M_w, num_mc)
#         cdf_term = np.log(norm.cdf(einsum_term) + 10e-10) # Shift away from 0
#         cdf_term_sub = np.log(1 - norm.cdf(einsum_term) + 10e-10) # Shift away from 0
                    
#         # Multiply and avergage to get mc approx
#         mc_approx = np.einsum('mpc,nmc->nmp', vec, cdf_term, optimize=True) / num_mc # (num_nodes, M_w, P)
#         mc_approx_sub = np.einsum('mpc,nmc->nmp', vec, cdf_term_sub, optimize=True) / num_mc
                
#         # Scale by matrix
#         scaled = np.einsum('mij,nmj->nmi', sigma_phi_inv_best, mc_approx, optimize=True) # (num_nodes, M_w, P)
#         scaled_sub = np.einsum('mij,nmj->nmi', sigma_phi_inv_best, mc_approx_sub, optimize=True)
                    
#         # Sum over phi_w
#         grad_theta += np.einsum('nm,nmp->mp', self.phi_w, scaled)
#         grad_theta += np.einsum('nm,nmp->mp',
#                                 (np.cumsum(self.phi_w[:,::-1], axis=1)[:, ::-1] - self.phi_w), 
#                                 scaled_sub)            
#         # Subtract remaining terms (CHANGED TO SUBTRACT)
#         for m in range(self.M_w):
#             grad_theta[m,:] -= ((self.nu_sigma2[m] / self.omega_sigma2[m]) * 
#                                 (theta_phi_best[m,:] - self.theta_phi0[m,:])
#             )
                    
#         # WE WANT TO MAXIMISE THE ELBO, SO APPLY ADAM TO THE NEGATIVE GRADIENT
#         grad_theta = -grad_theta
                                
#         ## Estimate the expectation for sigma
#         # Compute outer product and add matrices - need to rewrite in suffix notation
#         scaled_outer_prod = np.zeros((self.M_w, self.P, self.P, num_mc))
        
#         # outer_prod = np.zeros((self.M_w, self.P, self.P, num_mc))
#         # for m in range(self.M_w):
#         #     for mc in range(num_mc):
#         #         outer_prod[m,:,:, mc] = np.outer(vec[m,:,mc], vec[m,:,mc])
        
#         for m in range(self.M_w):
#             for mc in range(num_mc):
#                 scaled_outer_prod[m,:,:, mc] = (sigma_phi_inv_best[m,:,:] @ 
#                                                 np.outer(vec[m,:,mc], vec[m,:,mc]) @
#                                                 sigma_phi_inv_best[m,:,:] - sigma_phi_inv_best[m,:,:])
        
#         mc_approx = np.zeros((self.num_nodes, self.M_w, self.P, self.P))
#         mc_approx_sub = np.zeros((self.num_nodes, self.M_w, self.P, self.P))
        
#         for i in range(self.num_nodes):
#             for m in range(self.M_w):
#                 mc_approx[i, m, :, :] = 0.5 * np.sum(scaled_outer_prod[m, :, :, :] * 
#                                                     cdf_term[i, m, :], axis=-1) / num_mc
#                 mc_approx_sub[i, m, :, :] = 0.5 * np.sum(scaled_outer_prod[m, :, :, :] * 
#                                                         cdf_term_sub[i, m, :], axis=-1) / num_mc

#                 # # Scale by sigma_phi_inv and 1/2
#                 # mc_approx[i, m, :, :] = 0.5 * (
#                 #     sigma_phi_inv[m, :, :] @ mc_approx[i, m, :, :] @ sigma_phi_inv[m, :, :]
#                 #     - np.mean(cdf_term[i, m, :]) * sigma_phi_inv[m, :, :]
#                 # )
#                 # mc_approx_sub[i, m, :, :] = 0.5 * (
#                 #     sigma_phi_inv[m, :, :] @ mc_approx_sub[i, m, :, :] @ sigma_phi_inv[m, :, :]
#                 #     - np.mean(cdf_term_sub[i, m, :]) * sigma_phi_inv[m, :, :]
#                 # )
                            
#         # Sum over phi_w
#         for m in range(self.M_w):
#             for i in range(self.num_nodes):
#                 grad_sigma[m,:,:] += self.phi_w[i,m] * mc_approx[i,m,:,:]
        
#         for m in range(self.M_w):
#             for i in range(self.num_nodes):
#                 for k in range(m + 1, self.M_w):
#                     grad_sigma[m,:,:] += self.phi_w[i,k] * mc_approx_sub[i,m,:,:]

#         # Add on the remaining terms
#         for m in range(self.M_w):
#             grad_sigma += 0.5 * (
#                 -0.5 * self.nu_sigma2[m] / self.omega_sigma2[m] * np.eye(self.P) +
#                 0.5 * sigma_phi_inv_best[m,:,:]
#             )
                    
#         # # Add on the remaining terms
#         # for m in range(self.M_w):
#         #     grad_sigma += 0.5 * (
#         #         self.nu_sigma2[m] / self.omega_sigma2[m] * np.eye(self.P) +
#         #         sigma_phi_inv[m,:,:]
#         #     )
        
#         # Convert from sigma_m gradient to M_m gradient
#         for m in range(self.M_w):
#             grad_sigma_M_m = np.zeros(shape=(self.P, self.P))
#             L_m = _compute_L_from_M(M_best[m,:,:])
#             for i in range(self.P):
#                 for j in range(self.P):
#                     E = np.zeros(shape=(self.P, self.P))
#                     E[i,j] = 1
#                     if i == j:
#                         grad_sigma_M_m = np.exp(M_best[m,i,i]) * (E @ L_m.T + L_m @ E)
#                     elif i < j:
#                         grad_sigma_M_m = (E @ L_m.T + L_m @ E.T)
#                     else:
#                         pass
#                     grad_M[m,i,j] = np.trace(grad_sigma[m,:,:].T @ grad_sigma_M_m)
                    
#         # WE WANT TO MAXIMISE THE ELBO, SO APPLY ADAM TO THE NEGATIVE GRADIENT
#         grad_M = -grad_M
                        
#         # ADAM steps
#         for k in range(self.M_w):
#             first_moment_theta_curr[k] = beta1 * self.first_moment_theta_best[k] + (1 - beta1) * grad_theta[k]
#             second_moment_theta_curr[k] = beta2 * self.second_moment_theta_best[k] + (1 - beta2) * grad_theta[k] ** 2
#             first_moment_M_curr[k] = beta1 * self.first_moment_M_best[k] + (1 - beta1) * grad_M[k]
#             second_moment_M_curr[k] = beta2 * self.second_moment_M_best[k] + (1 - beta2) * grad_M[k] ** 2                

#             first_moment_theta_bias_curr[k] = first_moment_theta_curr[k] / (1 - beta1 ** self.step_theta[k])
#             second_moment_theta_bias_curr[k] = second_moment_theta_curr[k] / (1 - beta2 ** self.step_theta[k])
#             first_moment_M_bias_curr[k] = first_moment_M_curr[k] / (1 - beta1 ** self.step_sigma[k])
#             second_moment_M_bias_curr[k] = second_moment_M_curr[k] / (1 - beta2 ** self.step_sigma[k])
        
#         # Update the ELBO and assess the convergence
#         for k in range(int(2 * self.M_w)):
#             if k < self.M_w:
#                 ## Updating theta
#                 if not param_converged[k]:
#                     ELBO_alpha_test = np.zeros((len(alpha_set), ))
#                     theta_phi_curr = theta_phi_best.copy()
#                     for idx_test, alpha_test in enumerate(alpha_set):
#                         # Update a temporary version for ELBO evaluation
#                         theta_phi_curr[k] = (
#                             theta_phi_best[k] 
#                             - alpha_test * first_moment_theta_bias_curr[k] / (np.sqrt(second_moment_theta_bias_curr[k]) + eps)
#                         ).copy()
#                         # Compute the ELBO for that learning rate
#                         ELBO_alpha_test[idx_test] = (
#                             self._compute_phi_ELBO(theta_phi=theta_phi_curr, 
#                                                 sigma_phi=sigma_phi_best)
#                             )
#                     # Optimal update
#                     alpha_opt = float(alpha_set[np.argmax(ELBO_alpha_test)])
#                     theta_phi_curr[k] = (
#                             theta_phi_best[k] 
#                             - alpha_opt * first_moment_theta_bias_curr[k] / (np.sqrt(second_moment_theta_bias_curr[k]) + eps)
#                         ).copy() # only change the kth of the best to the current
#                     ELBO_curr = (
#                         self._compute_phi_ELBO(theta_phi=theta_phi_curr, 
#                                             sigma_phi=sigma_phi_best)
#                     )
#                     # if ELBO increases, update the parameters and ELBO, else move on
#                     if ELBO_curr > ELBO_best[k]:
#                         # ELBO has increased, so update the parameter value
#                         theta_phi_best[k] = theta_phi_curr[k].copy()
#                         self.first_moment_theta_best[k] = first_moment_theta_curr[k].copy()
#                         self.second_moment_theta_best[k] = second_moment_theta_curr[k].copy()
#                         # check relative increase
#                         if ((ELBO_curr - ELBO_best[k]) / np.abs(ELBO_best[k]) < eps_ELBO):
#                             param_converged[k] = True
#                             print(f"Parameter theta{k} converged.")
#                         ELBO_best[k] = ELBO_curr
#                     else:
#                         # ELBO has decreased, so stop updating (CAN MAKE THIS MORE ROBUST)
#                         ELBO_phi_dec_track[k] += 1
#                         if ELBO_phi_dec_track[k] == 3:
#                             param_converged[k] = True
#                             print(f"Parameter theta_{k} decreased.")

#             else:
#                 ## Updating sigma
#                 if not param_converged[k]:
#                     ELBO_alpha_test = np.zeros((len(alpha_set), ))
#                     sigma_phi_curr = sigma_phi_best.copy()
#                     sigma_phi_inv_curr = sigma_phi_inv_best.copy()
#                     M_curr = M_best.copy()
#                     for idx_test, alpha_test in enumerate(alpha_set):
#                         # Update a temporary version for ELBO evaluation
#                         M_curr[int(k - self.M_w)] = (
#                             M_best[int(k - self.M_w)] 
#                             - alpha_test * first_moment_M_bias_curr[int(k - self.M_w)] / 
#                             (np.sqrt(second_moment_M_bias_curr[int(k - self.M_w)]) + eps)
#                         ).copy()
#                         # Convert back to sigma_phi 
#                         sigma_phi_curr[int(k - self.M_w),:,:], sigma_phi_inv_curr[int(k - self.M_w),:,:] = (
#                             _compute_sigma_and_inv_from_M(M_curr[int(k - self.M_w)])
#                         )
#                         # Compute the ELBO for that learning rate
#                         ELBO_alpha_test[idx_test] = (
#                             self._compute_phi_ELBO(theta_phi=theta_phi_best, 
#                                                 sigma_phi=sigma_phi_curr)
#                             )
#                     # Optimal update
#                     alpha_opt = float(alpha_set[np.argmax(ELBO_alpha_test)])
#                     M_curr[int(k - self.M_w)] = (
#                             M_best[int(k - self.M_w)] 
#                             - alpha_opt * first_moment_M_bias_curr[int(k - self.M_w)] / 
#                             (np.sqrt(second_moment_M_bias_curr[int(k - self.M_w)]) + eps)
#                             ).copy() # only change the kth of the best to the current
#                     # Convert back to sigma_phi 
#                     sigma_phi_curr[int(k - self.M_w),:,:], sigma_phi_inv_curr[int(k - self.M_w),:,:] = (
#                         _compute_sigma_and_inv_from_M(M_curr[int(k - self.M_w)])
#                         )
#                     ELBO_curr = (
#                         self._compute_phi_ELBO(theta_phi=theta_phi_best, 
#                                             sigma_phi=sigma_phi_curr)
#                     )
#                     # if ELBO increases, update the parameters and ELBO, else move on
#                     if ELBO_curr > ELBO_best[k]:
#                         # ELBO has increased, so update the parameter value
#                         M_best[int(k - self.M_w)] = M_curr[int(k - self.M_w)].copy()
#                         sigma_phi_best[int(k - self.M_w)] = sigma_phi_curr[int(k - self.M_w)].copy()
#                         sigma_phi_inv_best[int(k - self.M_w)] = sigma_phi_inv_curr[int(k - self.M_w)].copy()
#                         self.first_moment_M_best[int(k - self.M_w)] = first_moment_M_curr[int(k - self.M_w)].copy()
#                         self.second_moment_M_best[int(k - self.M_w)] = second_moment_M_curr[int(k - self.M_w)].copy()
#                         # check relative increase
#                         if ((ELBO_curr - ELBO_best[k]) / np.abs(ELBO_best[k]) < eps_ELBO):
#                             param_converged[k] = True
#                             print(f"Parameter sigma_{k} converged.")
#                         ELBO_best[k] = ELBO_curr
#                     else:
#                         # ELBO has decreased, so stop updating (CAN MAKE THIS MORE ROBUST)
#                         ELBO_phi_dec_track[k] += 1
#                         if ELBO_phi_dec_track[k] == 3:
#                             param_converged[k] = True
#                             print(f"Parameter sigma_{k} decreased.")
                        
#         # Increase the step for ADAM for relevant parameters
#         self.step_theta[~param_converged[:int(self.M_w)]] += 1
#         self.step_sigma[~param_converged[int(self.M_w):]] += 1 
        
#     print(f"Stopped gradient updates at step: {step + 1}")
    
#     # Update the parameters
#     self.theta_phi = theta_phi_best.copy()
#     self.sigma_phi = sigma_phi_best.copy()
    
    
    
    
    

# for k in range(M_w):
#     ####
#     #~~ ADAM for theta_phi_k
#     ####
#     theta_phi
    
#     ####
#     #~~ ADAM for sigma_phi_k
#     ####
#     theta_phi_tensor = torch.tensor(theta_phi, dtype=torch.float32)
#     optimizer = torch.optim.Adam([theta_phi_tensor], lr=0.1)

#     # Parameters for tracking decreases
#     max_decreases = 3  # Maximum allowed consecutive decreases
#     decrease_counter = 0
#     best_theta = theta_phi_tensor.clone().detach()  # Store best parameters
#     best_ELBO = float('-inf')
#     no_improvement_counter = 0

#     # Optimization loop
#     for _ in range(max_grad_steps):
#         optimizer.zero_grad()
                    
#         # Evaluate ELBO
#         ELBO = compute_ELBO(np.asarray(theta_phi_tensor))
#         if _ == 0:
#             ELBO_prev = ELBO
#             best_ELBO = ELBO
        
#         # Check if ELBO improved
#         if ELBO > best_ELBO:
#             best_ELBO = ELBO
#             best_theta = theta_phi_tensor.clone().detach()
#             decrease_counter = 0  # Reset counter on improvement
#         else:
#             decrease_counter += 1
        
#         # If too many decreases, revert and break
#         if decrease_counter >= max_decreases:
#             print(f"Reverting to best parameters after {max_decreases} decreases")
#             theta_phi_tensor.data = best_theta.data  # Revert to best parameters
#             break
        
#         # Compute and apply gradient
#         gradient = compute_gradient(theta_phi)
#         theta_phi.grad = torch.tensor(gradient, dtype=torch.float32)
#         optimizer.step()
        
#         # Print progress
#         print(f"Param: {theta_phi.detach().numpy()}, ELBO: {ELBO}")
#         if _ > 0:
#             rel_change = (ELBO - ELBO_prev) / ELBO_prev
#             print(f"Relative change: {rel_change}")
#             if abs(rel_change) < 1e-6:
#                 break
#         ELBO_prev = ELBO
    
#     optimizer = torch.optim.Adam([theta_phi], lr=0.1)

#     # Optimization loop
#     for _ in range(100):  # Number of iterations
#         optimizer.zero_grad()

#         VB.theta_phi[0] = np.asarray(theta_phi)

#         # Evaluate the function (optional, if you need to track ELBO)
#         ELBO = compute_ELBO(VB.theta_phi)
#         if _ == 0:
#             ELBO_prev = ELBO

#         # Compute the gradient manually
#         gradient = compute_gradient(theta_phi)

#         # Attach the manually computed gradient to the parameter
#         theta_phi.grad = torch.tensor(gradient, dtype=torch.float32)

#         # Update parameters
#         optimizer.step()

#         # Print progress
#         print(f"Param: {theta_phi.detach().numpy()}, ELBO: {ELBO}")
#         if _ > 0:
#             print((ELBO - ELBO_prev) / ELBO_prev)
#             ELBO_prev = ELBO





# ### VERSION 1 ###
# def _update_q_phi(self, num_mc: int, max_num_grad_steps: int, CAVI_rep: int,
#                     alpha: float=0.001, beta1: float=0.9, beta2: float=0.999, 
#                     eps: float=10**-8, eps_ELBO: float=10**-3,
#                     alpha_set_theta: list=None, alpha_set_sigma: list=None,
#                     max_decrease_count: int=3):
#         """
#         A method for computing the variational approximation for each \phi_{k}. This
#         uses an ADAM optimiser to perform the gradient ascent steps to maximise the ELBO.
#         Parameters:
#             - num_mc: number of MC samples for expectation approximations.
#             - max_num_grad_steps: number of ADAM steps to compute.
#             - alpha, beta1, beta2, eps: standard ADAM parameters.
#             - eps_ELBO: threshold for deciding on convergence of ELBO.
#         """
#         def _compute_M_from_sigma(sigma):
#             """
#             """
#             L = np.linalg.cholesky(sigma)
            
#             M = np.tril(L, k=-1) 
#             np.fill_diagonal(M, np.log(np.diag(L)))
            
#             return M
        
#         def _compute_L_from_M(M):
#             """
#             """
#             L = np.tril(M, k=-1)
#             np.fill_diagonal(L, np.exp(np.diag(M)))
            
#             return L
        
#         def _compute_sigma_and_inv_from_M(M):
#             """
#             """
#             L = _compute_L_from_M(M)
            
#             sigma = L @ L.T
#             sigma_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
            
#             return sigma, sigma_inv

#         for k in range(self.M_w):
#             ## In the outer loop, we do updates for \theta_k and \Sigma_k 
#             ## as these are dependent.
            
#             # Initialise temporary parameters
#             theta_phi_best = self.theta_phi.copy()
#             theta_phi_curr = self.theta_phi.copy()
#             sigma_phi_best = self.sigma_phi.copy()
#             sigma_phi_curr = self.sigma_phi.copy()
#             sigma_phi_inv_best = np.zeros((self.M_w, self.P, self.P))
#             M_best = np.zeros((self.M_w, self.P, self.P))
#             for m in range(self.M_w): # RECODE - INEFFICIENT.
#                 M_best[m,:,:] = _compute_M_from_sigma(sigma_phi_best[m,:,:])
#                 _, sigma_phi_inv_best[m,:,:] = (
#                     _compute_sigma_and_inv_from_M(M_best[m,:,:])
#                     )
#             sigma_phi_inv_curr = sigma_phi_inv_best.copy()
#             M_curr = M_best.copy()
            
#             # Compute initial ELBO value
#             ELBO_best_k = self._compute_full_ELBO()
            
#             # ADAM parameters
#             first_moment_theta_curr = self.first_moment_theta_best.copy()[k]
#             first_moment_M_curr = self.first_moment_M_best.copy()[k]
#             second_moment_theta_curr = np.zeros((self.P))
#             second_moment_M_curr = np.zeros((self.P, self.P))
            
#             first_moment_theta_bias_curr = np.zeros((self.P))
#             second_moment_theta_bias_curr = np.zeros((self.P))
#             first_moment_M_bias_curr = np.zeros((self.P, self.P))
#             second_moment_M_bias_curr = np.zeros((self.P, self.P))
            
#             decrease_counter = 0
#             for step in range(max_num_grad_steps):
#                 print(f"...theta_{k} and sigma_{k} iteration {step + 1}...", end='\r')
#                 ## Do ADAM for the inner loop.
#                 # Empty arrays for gradients
#                 grad_theta_k = np.zeros((self.P))
#                 grad_sigma_k = np.zeros((self.P, self.P))
#                 grad_M_k = np.zeros((self.P, self.P))
                
#                 # Sample phi and compute delta using current variational parameter values
#                 phi_k_samps = self._sample_phi(num_mc, theta_phi_curr[k], sigma_phi_curr[k],
#                                                 single=True) # (P, num_mc)
#                 phi_k_samps = phi_k_samps.T
                
#                 #### 
#                 #~~ Estimate the expectation for theta_k
#                 ####
                
#                 # Compute the phi - theta
#                 vec = (phi_k_samps - np.tile(theta_phi_curr[k, :], (num_mc, 1))) # (num_mc, P)
                
#                 # Once with einsum
#                 einsum_term = np.einsum('np,cp->nc', self.features, phi_k_samps, optimize=True) # (num_nodes, num_mc)
#                 cdf_term = np.log(norm.cdf(einsum_term) + 10e-10) # Shift away from 0
#                 cdf_term_sub = np.log(1 - norm.cdf(einsum_term) + 10e-10) # Shift away from 0
                            
#                 # Multiply and avergage to get mc approx
#                 mc_approx = np.einsum('cp,nc->np', vec, cdf_term, optimize=True) / num_mc # (num_nodes, P)
#                 mc_approx_sub = np.einsum('cp,nc->np', vec, cdf_term_sub, optimize=True) / num_mc
                        
#                 # Scale by matrix
#                 scaled = np.einsum('ij,nj->ni', sigma_phi_inv_curr[k], mc_approx, optimize=True) # (num_nodes, P)
#                 scaled_sub = np.einsum('ij,nj->ni', sigma_phi_inv_curr[k], mc_approx_sub, optimize=True)
                            
#                 # Sum over phi_w
#                 grad_theta_k += np.einsum('n,np->p', self.phi_w[:,k], scaled)
#                 grad_theta_k += np.einsum('n,np->p',
#                                             (np.cumsum(self.phi_w[:,::-1], axis=1)[:, ::-1] - self.phi_w)[:,k], 
#                                             scaled_sub)            
#                 # Subtract remaining terms (CHANGED TO SUBTRACT)
#                 grad_theta_k -= ((self.nu_sigma2[k] / self.omega_sigma2[k]) * 
#                                         (theta_phi_curr[k,:] - self.theta_phi0[k,:]))
                            
#                 # We want to maximise the ELBO, so negate for ADAM
#                 grad_theta_k = -grad_theta_k
                
#                 #### 
#                 #~~ Estimate the expectation for Sigma_k
#                 ####
                
#                 # Compute outer product and add matrices - need to rewrite in suffix notation
#                 scaled_outer_prod = np.zeros((self.P, self.P, num_mc))
                
#                 for mc in range(num_mc):
#                     scaled_outer_prod[:,:,mc] = (sigma_phi_inv_curr[k,:,:] @ 
#                                                     np.outer(vec[mc,:], vec[mc,:]) @
#                                                     sigma_phi_inv_curr[k,:,:] - sigma_phi_inv_curr[k,:,:])
                
#                 mc_approx = np.zeros((self.num_nodes, self.P, self.P))
#                 mc_approx_sub = np.zeros((self.num_nodes, self.P, self.P))
                
#                 for i in range(self.num_nodes):
#                     mc_approx[i, :, :] = 0.5 * np.sum(scaled_outer_prod * 
#                                                         cdf_term[i, :], axis=-1) / num_mc # (num_nodes, P, P)
#                     mc_approx_sub[i, :, :] = 0.5 * np.sum(scaled_outer_prod * 
#                                                             cdf_term_sub[i, :], axis=-1) / num_mc
#                 # Sum over phi_w
#                 for i in range(self.num_nodes):
#                     grad_sigma_k += self.phi_w[i,k] * mc_approx[i,:,:]
                
#                 grad_sigma_k += np.einsum('n,npq->pq',
#                                             (np.cumsum(self.phi_w[:,::-1], axis=1)[:, ::-1] - self.phi_w)[:,k], 
#                                             mc_approx_sub)        

#                 # Add on the remaining terms
#                 grad_sigma_k += 0.5 * (
#                         -self.nu_sigma2[k] / self.omega_sigma2[k] * np.eye(self.P) +
#                         sigma_phi_inv_curr[k,:,:]
#                     )
                
#                 # Convert from Sigma_K gradient to M_k gradient
#                 grad_sigma_M_k = np.zeros(shape=(self.P, self.P))
#                 L_k = _compute_L_from_M(M_curr[k,:,:])
#                 for i in range(self.P):
#                     for j in range(self.P):
#                         E = np.zeros(shape=(self.P, self.P))
#                         E[i,j] = 1
#                         if i == j:
#                             grad_sigma_M_k = np.exp(M_curr[k,i,i]) * (E @ L_k.T + L_k @ E)
#                         elif i < j:
#                             grad_sigma_M_k = (E @ L_k.T + L_k @ E.T)
#                         else:
#                             pass
#                         grad_M_k[i,j] = np.trace(grad_sigma_k.T @ grad_sigma_M_k)
                            
#                 # We want to maximise ELBO, so negate for ADAM
#                 grad_M_k = -grad_M_k
                
#                 ####
#                 #~~ ADAM Steps
#                 ####
#                 # CHANGED THE CODE HERE TO ALLOW IT TO TAKE A NUMBER OF STEPS BEFORE REJECTING COMPLETELY
#                 first_moment_theta_curr = beta1 * first_moment_theta_curr + (1 - beta1) * grad_theta_k
#                 second_moment_theta_curr = beta2 * second_moment_theta_curr + (1 - beta2) * grad_theta_k ** 2
#                 first_moment_M_curr = beta1 * first_moment_M_curr + (1 - beta1) * grad_M_k
#                 second_moment_M_curr = beta2 * second_moment_M_curr + (1 - beta2) * grad_M_k ** 2                

#                 first_moment_theta_bias_curr = first_moment_theta_curr / (1 - beta1 ** self.step_theta[k])
#                 second_moment_theta_bias_curr = second_moment_theta_curr / (1 - beta2 ** self.step_theta[k])
#                 first_moment_M_bias_curr = first_moment_M_curr / (1 - beta1 ** self.step_sigma[k])
#                 second_moment_M_bias_curr = second_moment_M_curr / (1 - beta2 ** self.step_sigma[k])
                
#                 ## Select optimal value of alpha
#                 ELBO_eval_k = np.zeros((len(alpha_set_theta), len(alpha_set_sigma)))
#                 theta_phi_eval = theta_phi_curr.copy()
#                 M_eval = M_curr.copy()
#                 sigma_phi_eval = sigma_phi_curr.copy()
#                 for idx_theta, alpha_theta in enumerate(alpha_set_theta):
#                     for idx_sigma, alpha_sigma in enumerate(alpha_set_sigma):
#                         # Updating theta_k
#                         theta_phi_eval[k] = (
#                                     theta_phi_curr[k] 
#                                     - alpha_theta * first_moment_theta_bias_curr / (np.sqrt(second_moment_theta_bias_curr) + eps)
#                                 ).copy()  
#                         # Updating Sigma_k
#                         M_eval[k] = (
#                             M_curr[k] 
#                             - alpha_sigma * first_moment_M_bias_curr / (np.sqrt(second_moment_M_bias_curr) + eps)
#                             ).copy()
#                         # Convert back to sigma_phi 
#                         sigma_phi_eval[k], _ = (
#                             _compute_sigma_and_inv_from_M(M_eval[k])
#                             )
#                         ELBO_eval_k[idx_theta, idx_sigma] = (
#                                 self._compute_phi_ELBO(theta_phi=theta_phi_eval, 
#                                                        sigma_phi=sigma_phi_eval)
#                                 )
#                 ## Compute parameter updates
#                 best_idx = np.unravel_index(np.argmax(ELBO_eval_k), ELBO_eval_k.shape)
#                 alpha_theta = alpha_set_theta[best_idx[0]]
#                 alpha_sigma = alpha_set_sigma[best_idx[1]]
#                 theta_phi_curr[k] = (
#                             theta_phi_curr[k] 
#                             - alpha_theta * first_moment_theta_bias_curr / (np.sqrt(second_moment_theta_bias_curr) + eps)
#                         ).copy()  
#                 # Updating Sigma_k
#                 M_curr[k] = (
#                     M_curr[k] 
#                     - alpha_sigma * first_moment_M_bias_curr / (np.sqrt(second_moment_M_bias_curr) + eps)
#                     ).copy()
#                 # Convert back to sigma_phi 
#                 sigma_phi_curr[k], sigma_phi_inv_curr[k] = (
#                     _compute_sigma_and_inv_from_M(M_curr[k])
#                     )
#                 ELBO_curr_k = (
#                         self._compute_phi_ELBO(theta_phi=theta_phi_curr, 
#                                             sigma_phi=sigma_phi_curr)
#                         )
                
#                 ## Checking ELBO
#                 if ELBO_curr_k > ELBO_best_k:
#                     print("ELBO increased", end='\r')
#                     ELBO_best_k = ELBO_curr_k # Update the best ELBO
#                     decrease_counter = 0 # Set decreasing counter to 0
                    
#                     # ELBO increased so update best parameters
#                     theta_phi_best = theta_phi_curr.copy()
#                     M_best = M_curr.copy()
#                     sigma_phi_best = sigma_phi_curr.copy()
#                     sigma_phi_inv_best = sigma_phi_inv_curr.copy()
#                     self.first_moment_theta_best[k] = first_moment_theta_curr.copy()
#                     self.second_moment_theta_best[k] = second_moment_theta_curr.copy()
#                     self.first_moment_M_best[k] = first_moment_M_curr.copy()
#                     self.second_moment_M_best[k] = second_moment_M_curr.copy()
                    
#                     # Increase steps
#                     self.step_theta[k] += 1
#                     self.step_sigma[k] += 1
#                 else:
#                     print("ELBO decreased", end='\r')
#                     decrease_counter += 1
#                     # Increase steps
#                     self.step_theta[k] += 1
#                     self.step_sigma[k] += 1
#                     if decrease_counter == max_decrease_count:
#                         # Shift the number of steps back
#                         self.step_theta[k] -= int(decrease_counter)
#                         self.step_sigma[k] -= int(decrease_counter)
#                         print("Maximum number of decreases reached.", end='\r')
#                         break
                    
#             ## Update final values after ADAM steps
#             # Update the parameters and proceed to next global group
#             self.theta_phi = theta_phi_best.copy()
#             self.sigma_phi = sigma_phi_best.copy()
                
                
                
                
                
            