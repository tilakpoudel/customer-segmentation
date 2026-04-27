# src/comparisons.py
"""Logic for running alternative clustering models and comparing them."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass

@dataclass
class KMeansResult:
    """Container for K-Means results."""
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    silhouette_score: float

def run_kmeans(data: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeansResult:
    """
    Run K-Means clustering on the provided data.
    
    Args:
        data (np.ndarray): Input data of shape (n_customers, n_features).
        n_clusters (int): Number of clusters (k).
        random_state (int): Seed for reproducibility.
        
    Returns:
        KMeansResult: Object containing model outputs and metrics.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    
    # Calculate silhouette score
    score = silhouette_score(data, labels)
    
    return KMeansResult(
        labels=labels,
        centers=kmeans.cluster_centers_,
        inertia=kmeans.inertia_,
        silhouette_score=score
    )

def calculate_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Calculate simple percentage agreement between two labeling sets.
    Note: Labels might be permuted, so this is a naive metric.
    In production, one would use Adjusted Rand Index (ARI).
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels_a, labels_b)
