# customer_segmentation/src/insights.py
"""Business logic for interpreting clusters and generating insights."""

import numpy as np
import pandas as pd

from config import SEGMENT_LABELS
from src.utils import setup_logging

logger = setup_logging()

def label_clusters(centers: np.ndarray, feature_names: list[str]) -> list[dict]:
    """
    Map cluster centers to human-readable segment labels.
    
    Logic:
    1. For each feature, determine if a cluster is 'High' (True) or 'Low' (False).
    2. 'High' is defined as being above the median of all cluster centers for that feature.
    3. Note: For Recency, 'High' value (days) is 'Low' performance. 
       So we flip the logic: High Performance Recency = Low value.
    
    Args:
        centers (np.ndarray): Cluster centers (k, n_features).
        feature_names (list[str]): Names of features (Recency, Frequency, Monetary).
        
    Returns:
        list[dict]: Metadata for each cluster.
    """
    k = centers.shape[0]
    # Calculate medians for each feature across all centers
    medians = np.median(centers, axis=0)
    
    cluster_meta = []
    
    for i in range(k):
        center = centers[i]
        
        # Initialize (Recency, Frequency, Monetary) status
        # If a feature is missing, we default to True (Good) so it doesn't 
        # penalize the segment label.
        status = [True, True, True]
        
        for idx, name in enumerate(feature_names):
            if "Recency" in name:
                status[0] = (center[idx] <= medians[idx])
            elif "Frequency" in name:
                status[1] = (center[idx] >= medians[idx])
            elif "Monetary" in name:
                status[2] = (center[idx] >= medians[idx])
        
        # Convert to tuple for dictionary lookup
        key = tuple(status)
        meta = SEGMENT_LABELS.get(key, SEGMENT_LABELS[(False, False, False)]).copy()
        meta["center_values"] = center
        cluster_meta.append(meta)
        
    return cluster_meta

def get_ambiguous_customers(
    membership_matrix: np.ndarray,
    customer_ids: pd.Index,
    threshold: float,
) -> pd.DataFrame:
    """
    Find customers whose maximum membership degree is below a certain threshold.
    
    Args:
        membership_matrix (np.ndarray): (k, n_customers).
        customer_ids (pd.Index): IDs of customers.
        threshold (float): Ambiguity threshold.
        
    Returns:
        pd.DataFrame: Table of ambiguous customers.
    """
    max_membership = np.max(membership_matrix, axis=0)
    ambiguous_mask = max_membership < threshold
    
    ambiguous_df = pd.DataFrame({
        "CustomerID": customer_ids[ambiguous_mask],
        "Max Membership": max_membership[ambiguous_mask]
    })
    
    # Add membership for each cluster
    for i in range(membership_matrix.shape[0]):
        ambiguous_df[f"Cluster {i}"] = membership_matrix[i, ambiguous_mask]
        
    return ambiguous_df.sort_values("Max Membership")

def generate_business_summary(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    cluster_meta: list[dict],
    ambiguous_count: int,
) -> dict:
    """
    Generate high-level business metrics.
    
    Args:
        rfm_df (pd.DataFrame): RFM table.
        labels (np.ndarray): Cluster labels.
        cluster_meta (list[dict]): Cluster metadata.
        ambiguous_count (int): Count of ambiguous customers.
        
    Returns:
        dict: Summary metrics.
    """
    df = rfm_df.copy()
    df["Cluster"] = labels
    
    revenue_at_risk = 0.0
    champion_revenue = 0.0
    per_cluster_stats = []
    
    for i, meta in enumerate(cluster_meta):
        cluster_df = df[df["Cluster"] == i]
        total_monetary = cluster_df["Monetary"].sum()
        avg_recency = cluster_df["Recency"].mean()
        
        stats = {
            "label": meta["label"],
            "count": len(cluster_df),
            "revenue": total_monetary,
            "avg_recency": avg_recency,
            "emoji": meta["emoji"]
        }
        per_cluster_stats.append(stats)
        
        if meta["label"] == "Champions":
            champion_revenue = total_monetary
        if "At-Risk" in meta["label"] or "Lost" in meta["label"]:
            revenue_at_risk += total_monetary
            
    return {
        "revenue_at_risk": revenue_at_risk,
        "champion_revenue": champion_revenue,
        "ambiguous_count": ambiguous_count,
        "per_cluster_stats": per_cluster_stats
    }
