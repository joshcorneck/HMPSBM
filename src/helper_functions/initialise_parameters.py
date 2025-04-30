"""
A script containing all functions necessary for informed initialisation.
"""

import numpy as np
import hdbscan

from scipy.stats import norm
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import svd
from sklearn.metrics import confusion_matrix

##~~
# Functions for initialising phi_k
##~~

def fit_single_probit(X: np.array, y: np.array, method: str = 'L-BFGS-B'):
    """
    Fit a single probit regression.
    
    Parameters:
        - X: feature matrix (N x P)
        - y: probability vector (N)
        - method: method for the optimiser.
    """
    N, P = X.shape
    
    def objective(phi):
        return np.sum((y - norm.cdf(X @ phi))**2)
    
    def gradient(phi):
        pred = norm.cdf(X @ phi)
        grad = -2 * np.sum((y - pred)[:,None] * 
                          norm.pdf(X @ phi)[:,None] * X,
                          axis=0)
        return grad
    
    # Initialize randomly
    phi_init = np.random.randn(P)
    
    result = minimize(
        objective,
        phi_init,
        method=method,
        jac=gradient
    )
    
    return result.x

def initialise_theta_phi(X: np.array, taus: np.array):
    """
    """
    num_nodes, P = X.shape
    M_w = taus.shape[1]
    phis = []
    
    def objective(phi_flat):
        """
        Compute the cross-entropy loss.
        """
        phis = phi_flat.reshape(-1, P)
        loss = 0
        products = np.ones(num_nodes)
        
        for k in range(M_w):
            pred_probs = norm.cdf(X @ phis[k]) * products
            loss -= np.sum(taus[:,k] * np.log(pred_probs + 1e-10))
            products *= (1 - norm.cdf(X @ phis[k]))
        
        return loss
    
    def gradient(phi_flat):
        """
        Compute the gradient of the cross-entropy loss.
        """
        phis = phi_flat.reshape(-1, P)
        grad = np.zeros_like(phi_flat)
        
        for i in range(num_nodes):
            for k in range(M_w):
                # Compute gradient for phi_k
                term1 = -(taus[i,k] * 
                        (norm.pdf(X[i] @ phis[k]) * X[i]) / 
                        (norm.cdf(X[i] @ phis[k]) + 10e-10)
                        )
                # Sum over previous categories
                term2 = 0
                for l in range(k + 1, M_w):
                    term2 -= (taus[i,l] *
                        (-norm.pdf(X[i] @ phis[k]) * X[i]) / 
                        ((1 - norm.cdf(X[i] @ phis[k])) + 10e-10)
                        )
                    
                grad[k*P:(k+1)*P] += term1 + term2
        
        return grad
    
    # Initial guess
    phi_init = np.random.randn(M_w * P)
    
    # Optimize
    result = minimize(
        objective,
        phi_init,
        method='L-BFGS-B',
        jac=gradient
    )
    
    return result.x.reshape(-1, P)

##~~
# Functions for initialising phi_w and phi_u
##~~
def assign_clusters(X: np.array, M: int):
    """
    Runs HDBSCAN to cluster X into a maximum of M clusters.
    
    Parameters:
        - X: array to be clustered.
        - M: maximum number of clusters.
        
    Output:
        - num_clusters: number of clusters assigned.
        - clustering.labels_: assignment array.
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

def assign_outliers(outlier_idxs: np.array, clustering: np.array, X: np.array, 
                    num_clusters: int):
    """
    Assign the outliers from the HDBSCAN to a cluster.
    
    Parameters: 
        - outliers_idxs: array of indices of outliers.
        - clustering: array of the assignments.
        - X: array of what was used to cluster.
        - num_clusters: how many clusters we want.
        
    Output:
        - outlier_assignments: array of clusters for outliers.
    """
    # Number of features used to embed.
    _, k_max = X.shape
    
    # Number of outliers
    num_outliers = outlier_idxs.sum()
    
    # Assign those with a -1 label to the group to whose centroid the
    # outlier is cloest to
    cluster_centroids = np.zeros((num_clusters, k_max))
    l2_distances = np.zeros((num_outliers, num_clusters))
    for cluster in range(num_clusters):
        cluster_idxs = (clustering == cluster)
        cluster_centroid = X[cluster_idxs, :].mean(axis=0)
        cluster_centroids[cluster, :] = cluster_centroid 
        l2_distances[:,cluster] = np.linalg.norm(
            X[outlier_idxs,:] - cluster_centroid, axis=1
        )
    outlier_assignments = np.argmin(l2_distances, axis=1)
    
    return outlier_assignments

def align_labels(ref_labels: np.array, other_labels: np.array):
    """
    Align `other_labels` to `ref_labels` by finding the best permutation.
    Returns the permuted version of `other_labels`.
    
    Parameters:
        - ref_labels: the labels of the first layer, used as reference.
        - other_labels: the labels of another layer to align.
    
    Output:
        - aligned_labels: aligned array of other_labels.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(ref_labels, other_labels)
    
    # Use Hungarian algorithm to maximize alignment
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    
    # Create a mapping from original labels to aligned labels
    label_mapping = {old: new for old, new in zip(col_ind, row_ind)}
    aligned_labels = np.vectorize(label_mapping.get)(other_labels)
    
    return aligned_labels

def initialise_phi_u(num_nodes: int, num_layers: int, k_max: int,
                     M_u: int, adj_tensor: np.array):
    """
    Initialise the array phi_u using spectral embedding of the layers.
    
    Parameters:
        - num_nodes: number of nodes in the network.
        - num_layers: number of layers in the network.
        - k_max: embedding dimension.
        - M_u: truncation of layer-level groups.
        - adj_tensor: adjacency tensor of the network.
    
    Output:
        - phi_u: initialised phi_u array.
    """
    # Empty array for phi_u
    phi_u = np.zeros((num_layers, num_nodes, M_u))
    
    # Store the embeddings
    left_embed = np.zeros((num_nodes, num_layers, k_max))

    # Layer clusterings
    layer_left_clusterings = np.zeros((num_nodes, num_layers))

    #####
    ##~~ Estimate phi_u ~~##
    #####    
    # Embed each layer
    for l in range(num_layers):
        A_l = adj_tensor[l,:,:].copy()
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
        num_clusters, layer_left_clusterings[:,l] = assign_clusters(X, M_u)
        
        # Assign those with a -1
        outlier_idxs = layer_left_clusterings[:,l] == -1
        
        if outlier_idxs.sum() == num_nodes:
            layer_left_clusterings[:,l] = 0
        elif outlier_idxs.sum() != 0:
            layer_left_clusterings[outlier_idxs, l] = (
                assign_outliers(outlier_idxs, 
                                layer_left_clusterings[:,l],
                                X, num_clusters)
                )
                    
    # Initiliase phi_u for 0th layer
    phi_u[0,np.arange(num_nodes),
          layer_left_clusterings[:,0].astype(int)] = 1
    
    # Adjust the enumeration to maximise the similarity between layers
    labels_base = layer_left_clusterings[:,0].astype(int) # Base labels
    for l in range(1, num_layers):
        aligned_labels = align_labels(
            labels_base, 
            layer_left_clusterings[:,l])
        
        # Initiliase phi_u for that layer
        phi_u[l,np.arange(num_nodes),
              aligned_labels.astype(int)] = 1
        
    return phi_u
     
def initialise_phi_w(num_nodes: int, num_layers: int, k_max: int,
                     M_w: int, features: np.array, adj_tensor: np.array):
    """
    Initialise the array phi_u using spectral embedding of the layers.
    
    Parameters:
        - num_nodes: number of nodes in the network.
        - num_layers: number of layers in the network.
        - k_max: embedding dimension.
        - M_w: truncation of global groups.
        - features: array of node-level features.
        - adj_tensor: adjacency tensor of the network.
    
    Output:
        - phi_w: initialised phi_w array.
    """
    # Empty array for phi_w
    phi_w = np.zeros((num_nodes, M_w))
    
    # Cluster the features
    num_clusters, global_clusterings = assign_clusters(features, M_w)
    
    # Assign those with a -1 label
    outlier_idxs = global_clusterings == -1
    if outlier_idxs.sum() != 0:
        global_clusterings[outlier_idxs] = (
            assign_outliers(outlier_idxs, global_clusterings,
                            features, num_clusters)
        )
    
    phi_w[np.arange(num_nodes), global_clusterings] = 1
    
    return phi_w
    
    
def initialise_alpha_beta_rho(num_layers: int, M_u: int, adj_tensor: np.array,
                              phi_u: np.array, num_adj_samps: int):
    """
    Initiliase alpha_rho and beta_rho.
    
    Parameters:
        - num_layers: number of layers in the network.
        - M_u: truncation of layer-level groups.
        - adj_tensor: adjacency tensor of the network.
        - phi_u: initialised value fo phi_u.
        
    Output:
        - alpha_rho, beta_rho: initialised estimates.
    """
    # Empty array for connectivity estimate    
    rho_estimate = np.ones((num_layers, M_u, M_u)) * 10e-5
    for l in range(num_layers):
        adj_l = adj_tensor[l].copy()
        for k in range(M_u):
            # Get indices of nodes with latent group k
            idxs_k = np.where(phi_u[l].argmax(axis=1) == k)[0]
            if idxs_k.size == 0:
                continue
            for m in range(M_u):
                if k == m:
                    adj_l_k_to_k = adj_l[idxs_k, idxs_k]
                    rho_estimate[l,k,k] = adj_l_k_to_k.mean() / num_adj_samps
                else:
                    # Get indices of nodes with latent group m
                    idxs_m = np.where(phi_u[l].argmax(axis=1) == m)[0]
                    if idxs_m.size == 0:
                        continue
                    adj_l_k_to_m = adj_l[np.ix_(idxs_k, idxs_m)]
                    rho_estimate[l,k,m] = adj_l_k_to_m.mean() / num_adj_samps     
    rho_estimate = rho_estimate.mean(axis=0)

    # There will be an error if rho == 1
    idxs_ones = np.where(rho_estimate == 1)
    rho_estimate[np.ix_(idxs_ones[0], idxs_ones[1])] -= 10e-5
    
    # Compute alpha_rho and beta_rho by setting beta_rho = 1
    beta_rho = np.ones((M_u, M_u))
    alpha_rho = beta_rho * rho_estimate / (1 - rho_estimate)
    
    return alpha_rho, beta_rho

##~~
# Functions for initialising alpha_gamma and beta_gamma
##~~
def initialise_alpha_beta_gamma(adj_tensor: np.array, phi_w: np.array,
                                phi_u: np.array):
    """
    Initiliase alpha_gamma and beta_gamma.
    
    Parameters:
        - adj_tensor: adjacency tensor of the network.
        - phi_w: initialised value fo phi_w.
        - phi_u: initialised value fo phi_u.
        
    Output:
        - alpha_gamma, beta_gamma: initialised estimates.
    """
    num_layers, _, M_u = phi_u.shape
    _, M_w = phi_w.shape
    
    alpha_gamma = np.zeros((M_w, M_u))
    beta_gamma = np.ones((M_w, M_u))
        
    for k in range(M_w):
        # Indices of nodes in global group k
        group_k_nodes = np.argwhere(phi_w.argmax(axis=1) == k).flatten()
        
        # Layer-level assignments for nodes i global group k
        layer_level_groups_k = phi_u[:, group_k_nodes, :].argmax(axis=2).flatten()
        
        # If no nodes in global group k
        if (len(layer_level_groups_k) == 0):
            alpha_gamma[k,:] = 1 / (M_u - 1) # makes the mean 1/M_u (equal spread)
        # Otherwise
        else:
            for m in range(M_u):
                gamma_km = (layer_level_groups_k == m).sum() / len(layer_level_groups_k)
                if gamma_km == 1:
                    alpha_gamma[k,m] = 1
                    beta_gamma[k,m] = 10e-3
                elif gamma_km == 0:
                    alpha_gamma[k,m] = 10e-3
                else:
                    alpha_gamma[k,m] = gamma_km / (1 - gamma_km)   
                        
    return alpha_gamma, beta_gamma
        