# from kmeans import kmeans_equal
from balanced_kmeans import kmeans_equal
import torch

N = 8
batch_size = 2
num_clusters = 4

cluster_size = N // num_clusters
X = torch.randn(batch_size, N, 3)
print(X)
choices, centers = kmeans_equal(X, num_clusters, cluster_size)

print(choices[0].bincount(minlength=num_clusters))