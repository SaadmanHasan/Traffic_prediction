import numpy as np
from sklearn.cluster import KMeans

def equal_size_kmeans(X, k):
    points = X.shape[0]
    size = points // k
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    cluster_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    while(True):
        cluster_sizes = np.zeros(k, dtype=int)
        pairwise_distance = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        for i in range(points):
            cluster_assignments[i] = np.argmax(pairwise_distance[i])
            cluster_sizes[cluster_assignments[i]] += 1
        
        excess_indices = np.where(cluster_sizes > size)[0]
        less_indices = np.where(cluster_sizes < size)[0]

        while len(excess_indices) > 0:
            for i in excess_indices:
                all_indices = np.where(cluster_assignments == i)[0]
                distances = pairwise_distance[all_indices,i]
                sorted_indices = np.argsort(distances)
                all_indices = all_indices[sorted_indices]
                num_to_move = int(cluster_sizes[i] - size)
                indices_to_move = all_indices[-num_to_move:]
                for j in indices_to_move:
                    insert = np.argmax(pairwise_distance[j, less_indices])
                    cluster_assignments[j] = less_indices[insert]
                    cluster_sizes[i] -= 1
                    cluster_sizes[less_indices[insert]] += 1

            excess_indices = np.where(cluster_sizes > size)[0]
            less_indices = np.where(cluster_sizes < size)[0]
        
        if len(excess_indices) == 0 and len(less_indices) == 0:
            break
    
    return cluster_assignments, centroids
