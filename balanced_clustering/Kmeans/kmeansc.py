import numpy as np
from balanced_kmeans import kmeans_equal
import torch

def better_equal_size_kmeans(X, k=65, cluster_size=5):
    print("Input shape:", X.shape)
    
    # Ensure the input X is a numpy array
    if not isinstance(X, np.ndarray):
        raise ValueError("Input X must be a numpy array.")
    
    # Check dimensions
    n_samples, n_features = X.shape
    if n_samples != k * cluster_size:
        raise ValueError("The number of samples must be equal to k * cluster_size.")

    # Determine the device for tensor operations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert the input to a torch tensor and ensure it's float type
    X_tensor = torch.tensor(X, device=device).float()
    
    # Debug: print tensor information
    print("Tensor shape:", X_tensor.shape)
    print("Device:", device)

    # Perform balanced k-means clustering
    try:
        choices, _ = kmeans_equal(X_tensor, num_clusters=k, cluster_size=cluster_size)
    except RuntimeError as e:
        print("Runtime Error during kmeans_equal:", e)
        raise

    return choices

# Test the function
X = np.random.rand(65 * 5, 65)
choices = better_equal_size_kmeans(X, k=65, cluster_size=5)
print("Choices shape:", choices.shape)
print("Choices:", choices)

def equal_size_kmeans(X, k, runs=3, spectral=0):
    points = X.shape[0]
    size = points / k

    centroids = np.random.rand(k, X.shape[1])

    for _ in range(runs):
        cluster_assignments = np.full(X.shape[0], -1) 
        cluster_sizes = np.zeros(k, dtype=int)

        for i in range(points):
            cluster_assignments[i] = np.argmax(np.matmul(X[i], centroids.T))
            cluster_sizes[cluster_assignments[i]] += 1
        
        for j in range(k):
            while cluster_sizes[j] > size:
                excess_indices = np.where(cluster_assignments == j)[0]
                if len(excess_indices) == 0:
                    break
                excess_index = excess_indices[-1] 
                cluster_sizes[j] -= 1    

                for m in range(k):
                    if cluster_sizes[m] < size:
                        cluster_assignments[excess_index] = m
                        cluster_sizes[m] += 1
                        break
                        
        if not spectral:
            for i in range(k):
                centroids[i] = np.mean(X[cluster_assignments == i].numpy(), axis = 0, keepdims= True)

        else:
            for i in range(k):
                centroids[i] = np.mean(X[cluster_assignments == i], axis = 0, keepdims= True)

    return cluster_assignments