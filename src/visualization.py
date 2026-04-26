# src/visualization.py
"""Plotly-based visualization builders for the application."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import COLOR_PALETTE, PLOTLY_TEMPLATE

def plot_scatter_clusters(
    rfm_df: pd.DataFrame, 
    labels: np.ndarray, 
    membership_matrix: np.ndarray, 
    feature_x: str, 
    feature_y: str
) -> go.Figure:
    """Create a 2D scatter plot of clusters with membership-based sizing."""
    df_plot = rfm_df.copy()
    df_plot["Cluster"] = [f"Cluster {i}" for i in labels]
    df_plot["Max Membership"] = np.max(membership_matrix, axis=0)
    
    fig = px.scatter(
        df_plot,
        x=feature_x,
        y=feature_y,
        color="Cluster",
        size="Max Membership",
        opacity=0.7,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_PALETTE,
        title=f"{feature_x} vs {feature_y} by Cluster",
        hover_data=df_plot.columns
    )
    return fig

def plot_membership_heatmap(
    membership_matrix: np.ndarray, 
    customer_ids: pd.Index, 
    cluster_labels: list[str]
) -> go.Figure:
    """Create a heatmap of membership degrees for top 50 customers."""
    # Take first 50 for readability
    subset_u = membership_matrix[:, :50]
    subset_ids = [str(cid) for cid in customer_ids[:50]]
    
    fig = go.Figure(data=go.Heatmap(
        z=subset_u,
        x=subset_ids,
        y=cluster_labels,
        colorscale="Viridis",
        colorbar=dict(title="Membership")
    ))
    fig.update_layout(
        title="Membership Degrees (First 50 Customers)",
        xaxis_title="Customer ID",
        yaxis_title="Cluster",
        template=PLOTLY_TEMPLATE
    )
    return fig

def plot_membership_distribution(
    membership_matrix: np.ndarray, 
    cluster_labels: list[str]
) -> go.Figure:
    """Create a violin plot showing membership distribution per cluster."""
    data = []
    for i, label in enumerate(cluster_labels):
        for val in membership_matrix[i, :]:
            data.append({"Cluster": label, "Membership": val})
    
    df_plot = pd.DataFrame(data)
    fig = px.violin(
        df_plot,
        x="Cluster",
        y="Membership",
        color="Cluster",
        box=True,
        points="all",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_PALETTE,
        title="Distribution of Membership Degrees per Cluster"
    )
    return fig

def plot_cluster_bar(labels: np.ndarray, cluster_labels: list[str]) -> go.Figure:
    """Create a bar chart showing the count of customers per cluster."""
    counts = pd.Series(labels).value_counts().sort_index()
    df_plot = pd.DataFrame({
        "Cluster": cluster_labels,
        "Count": [counts.get(i, 0) for i in range(len(cluster_labels))]
    })
    
    fig = px.bar(
        df_plot,
        x="Cluster",
        y="Count",
        color="Cluster",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_PALETTE,
        title="Number of Customers per Cluster",
        text_auto=True
    )
    return fig

def plot_rfm_distributions(rfm_df: pd.DataFrame) -> go.Figure:
    """Create histograms for RFM features."""
    fig = go.Figure()
    for i, col in enumerate(rfm_df.columns):
        fig.add_trace(go.Histogram(
            x=rfm_df[col], 
            name=col,
            marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ))
    
    fig.update_layout(
        barmode="overlay",
        title="RFM Feature Distributions",
        template=PLOTLY_TEMPLATE,
        xaxis_title="Value",
        yaxis_title="Frequency"
    )
    fig.update_traces(opacity=0.75)
    return fig
