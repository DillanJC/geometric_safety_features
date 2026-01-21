import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_knn_uncertainty_derivation(X_train, X_query, k=20, eps=1e-8):
    """
    Manual derivation of k-NN uncertainty: U = (1/k) * ∑ ||q - n_i|| (mean distance)
    D = 1 / (min_distance + eps) (inverse local density)
    final_uncertainty = U / D = U * (min_distance + eps)
    
    Justification: Scales dispersion by sparsity for balanced uncertainty.
    """
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_train)
    dists, _ = nn.kneighbors(X_query, return_distance=True)
    
    # U: Mean distance to k neighbors
    U = np.mean(dists, axis=1)
    
    # min_distance: Distance to nearest neighbor
    min_d = dists[:, 0]
    
    # D: Inverse density (1 / (min_d + eps))
    D = 1 / (min_d + eps)
    
    # Final uncertainty: U / D = U * (min_d + eps)
    uncertainty = U / D
    
    return uncertainty, U, min_d
