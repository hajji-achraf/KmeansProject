import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Définition de la classe KMeans
class KMeans:
    def __init__(self, n_clusters, max_iter=100, distance_metric='euclidean'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric

    def fit(self, X):
        if self.n_clusters > X.shape[0]:
            raise ValueError("Le nombre de clusters ne peut pas être supérieur au nombre de points dans les données.")

        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            if self.distance_metric == 'euclidean':
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            elif self.distance_metric == 'manhattan':
                distances = np.sum(np.abs(X[:, np.newaxis] - centroids), axis=2)
            else:
                raise ValueError("Distance metric not supported.")

            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(new_centroids == centroids):
                break
            centroids = new_centroids
        self.cluster_centers_ = centroids
        self.labels_ = labels + 1  # Ajouter 1 à tous les labels pour commencer à partir de 1
        return centroids, self.labels_
