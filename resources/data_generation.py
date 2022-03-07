"""
Question 2: Part A
Data Generation
"""

# import make blobs and make moons that provides random data points in clusters. 
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt

# generate random linearly separable points using make_blobs
def generate_random_datapoints(samples, features=2, centers=2):
    (X, y) = make_blobs(n_samples=samples, n_features=features, centers=centers, cluster_std=1.3, random_state=1)
    y = y.reshape((y.shape[0], 1))
    return X, y

# generate random non linearly separable points using make_moons
def generate_random_datapoints_NS(samples):
    X, y = make_moons(samples, noise=0.1)
    y = y.reshape((y.shape[0], 1))
    return X, y

# plot the points 
def plot_points(X, y):
    plt.title("Data")
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y[:, 0], s=30)