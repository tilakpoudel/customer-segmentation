# config.py
"""Configuration constants for the Customer Segmentation application."""

import pathlib

# Paths
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "online_retail.xlsx"

# Clustering Parameters
CLUSTERING = {
    "default_k": 4,
    "default_m": 2.0,
    "error_tolerance": 1e-5,
    "k_max": 8,
    "k_min": 2,
    "m_max": 4.0,
    "m_min": 1.1,
    "max_iter": 150,
    "random_state": 42,
}

# RFM Parameters
RFM = {
    "frequency_col": "Frequency",
    "log_transform": True,
    "monetary_col": "Monetary",
    "recency_col": "Recency",
}

# Segment Labels and Business Logic
# Mapping: (High Recency, High Frequency, High Monetary) -> Segment Meta
# "High" here means "Good" (e.g., Low Recency days, High Frequency count)
SEGMENT_LABELS = {
    (True, True, True): {
        "color": "#2ECC71",
        "emoji": "🏆",
        "label": "Champions",
        "recommendations": "Reward them. Can be early adopters for new products. Will promote your brand.",
    },
    (True, True, False): {
        "color": "#27AE60",
        "emoji": "🤝",
        "label": "Loyal Customers",
        "recommendations": "Upsell higher value products. Ask for reviews. Engaged but low spend.",
    },
    (True, False, True): {
        "color": "#3498DB",
        "emoji": "🌱",
        "label": "Potential Loyalists",
        "recommendations": "Offer loyalty program. Recommend other products to increase frequency.",
    },
    (True, False, False): {
        "color": "#5DADE2",
        "emoji": "🆕",
        "label": "New Customers",
        "recommendations": "Provide onboarding support. Give them a reason to come back soon.",
    },
    (False, True, True): {
        "color": "#E74C3C",
        "emoji": "⚠️",
        "label": "At-Risk High Value",
        "recommendations": "Send personalized emails with big discounts to reconnect. Do not lose them.",
    },
    (False, True, False): {
        "color": "#E67E22",
        "emoji": "😴",
        "label": "About to Sleep",
        "recommendations": "Share valuable resources/content. Offer limited-time discounts to reactivate.",
    },
    (False, False, True): {
        "color": "#F1C40F",
        "emoji": "💰",
        "label": "Promising",
        "recommendations": "Re-engage with high-value offers. Investigate why they stopped buying.",
    },
    (False, False, False): {
        "color": "#95A5A6",
        "emoji": "💤",
        "label": "Lost Customers",
        "recommendations": "Revival campaign if possible, otherwise ignore to focus on better segments.",
    },
}

# Membership Analysis
AMBIGUITY_THRESHOLD = 0.55

# UI Aesthetics
PAGE_CONFIG = {
    "layout": "wide",
    "page_icon": "🎯",
    "page_title": "Customer Segmentation | Fuzzy C-Means",
}

PLOTLY_TEMPLATE = "plotly_white"
COLOR_PALETTE = ["#2ECC71", "#3498DB", "#E74C3C", "#F1C40F", "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]
