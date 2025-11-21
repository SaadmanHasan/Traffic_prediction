import numpy as np
from sklearn.metrics import pairwise_kernels
from model.kmeansc import equal_size_kmeans, better_equal_size_kmeans

def balanced_spectral_clustering(X, k, gamma=1.0):
    similarity_matrix = pairwise_kernels(X, metric='rbf', gamma=gamma)
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    eigenvectors = np.array(eigenvectors)

    return equal_size_kmeans(eigenvectors[:, :k], k, spectral=1)
    # return better_equal_size_kmeans(eigenvectors[:, :k], k)