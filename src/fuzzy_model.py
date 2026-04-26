# /src/fuzzy_model.py
"""Fuzzy C-Means clustering implementation and prediction logic."""

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score

from src.utils import setup_logging

logger = setup_logging()

@dataclass
class FuzzyResult:
    """Container for Fuzzy C-Means results."""
    membership_matrix: np.ndarray   # shape (k, n_customers)
    centers: np.ndarray             # shape (k, n_features)
    labels: np.ndarray              # argmax cluster per customer
    partition_coefficient: float    # fuzzy evaluation metric
    silhouette_score: float

def run_fuzzy_cmeans(
    data: np.ndarray,
    n_clusters: int,
    fuzziness: float,
    max_iter: int,
    error: float,
    random_state: int = 42,
) -> FuzzyResult:
    """
    Run skfuzzy.cmeans on the provided data.
    
    Args:
        data (np.ndarray): Input data of shape (n_customers, n_features).
        n_clusters (int): Number of clusters (k).
        fuzziness (float): Fuzziness parameter (m).
        max_iter (int): Maximum number of iterations.
        error (float): Error tolerance for convergence.
        random_state (int): Seed for reproducibility.
        
    Returns:
        FuzzyResult: Object containing model outputs and metrics.
    """
    logger.info(f"Running Fuzzy C-Means with k={n_clusters}, m={fuzziness}")
    
    # skfuzzy.cmeans expects data in shape (n_features, n_customers)
    data_t = data.T
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_t,
        c=n_clusters,
        m=fuzziness,
        error=error,
        maxiter=max_iter,
        seed=random_state
    )
    
    # Get hard labels for metrics and visualization
    labels = np.argmax(u, axis=0)
    
    # Calculate silhouette score (requires hard labels)
    # Note: If n_clusters=1, silhouette is not defined, but k_min is 2.
    score = silhouette_score(data, labels)
    
    return FuzzyResult(
        membership_matrix=u,
        centers=cntr,
        labels=labels,
        partition_coefficient=fpc,
        silhouette_score=score
    )

def predict_new_customer(
    rfm_values: np.ndarray,
    trained_centers: np.ndarray,
    fuzziness: float,
) -> np.ndarray:
    """
    Predict cluster memberships for a new customer.
    
    Args:
        rfm_values (np.ndarray): Normalized RFM values (1, 3).
        trained_centers (np.ndarray): Centers from trained model (k, 3).
        fuzziness (float): Fuzziness parameter (m).
        
    Returns:
        np.ndarray: Membership scores for each cluster.
    """
    # skfuzzy.cmeans_predict expects centers in shape (k, n_features)
    # and data in shape (n_features, n_samples)
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        rfm_values.T,
        trained_centers,
        m=fuzziness,
        error=1e-5,
        maxiter=150
    )
    return u.flatten()
