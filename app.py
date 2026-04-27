# app.py
"""Streamlit entry point for the Customer Segmentation application."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

from config import (
    AMBIGUITY_THRESHOLD,
    CLUSTERING,
    PAGE_CONFIG,
    RFM as RFM_CONFIG
)
from src.fuzzy_model import run_fuzzy_cmeans, predict_new_customer
from src.insights import (
    generate_business_summary,
    get_ambiguous_customers,
    label_clusters
)
from src.preprocessing import clean_data, load_data, normalize_rfm
from src.rfm import compute_rfm
from src.utils import format_currency, format_number, setup_logging
from src.visualization import (
    plot_cluster_bar,
    plot_membership_distribution,
    plot_membership_heatmap,
    plot_rfm_distributions,
    plot_scatter_clusters,
    plot_model_comparison
)
from src.comparisons import run_kmeans, calculate_agreement

# Initialize logging
logger = setup_logging()

# Page configuration
st.set_page_config(**PAGE_CONFIG)

REQUIRED_COLUMNS = ["CustomerID", "InvoiceNo", "Quantity", "UnitPrice", "InvoiceDate"]

def validate_columns(df: pd.DataFrame) -> bool:
    """Check if all required columns are present in the dataframe."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        st.info(f"The file must contain: {', '.join(REQUIRED_COLUMNS)}")
        return False
    return True

def get_template_df() -> pd.DataFrame:
    """Create a dummy dataframe for the user to use as a template."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS + ["StockCode", "Description"])

def get_sample_data() -> pd.DataFrame:
    """Generate sample RFM-like transactional data for demonstration."""
    np.random.seed(42)
    n_rows = 1000
    customer_ids = np.random.randint(10000, 15000, 200)
    
    data = {
        "CustomerID": np.random.choice(customer_ids, n_rows),
        "InvoiceNo": np.random.randint(500000, 600000, n_rows),
        "StockCode": np.random.randint(20000, 30000, n_rows).astype(str), # Cast to string for Arrow compatibility
        "Quantity": np.random.randint(1, 50, n_rows),
        "UnitPrice": np.random.uniform(1.0, 100.0, n_rows),
        "InvoiceDate": pd.date_range(start="2023-01-01", end="2023-12-31", periods=n_rows)
    }
    return pd.DataFrame(data)

@st.cache_data
def cached_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate cleaning and RFM computation with caching."""
    cleaned_df = clean_data(df)
    rfm_df = compute_rfm(cleaned_df)
    return rfm_df

@st.cache_resource
def cached_model_execution(data: np.ndarray, params: dict):
    """Run fuzzy clustering and cache the result object."""
    return run_fuzzy_cmeans(data, **params)

@st.cache_resource
def cached_kmeans_execution(data: np.ndarray, k: int):
    """Run k-means clustering and cache the result object."""
    return run_kmeans(data, k)

# --- Sidebar ---
st.sidebar.title("🛠️ Control Panel")

data_source = st.sidebar.radio("Data Source", ["Upload File", "Use Sample Data"])

# Sidebar Download Template (Displayed above uploader)
with st.sidebar.expander("📝 Data Format Help", expanded=False):
    st.write("Your file must include these columns:")
    for col in REQUIRED_COLUMNS:
        st.code(col)
    
    template_csv = get_template_df().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Template",
        data=template_csv,
        file_name="rfm_template.csv",
        mime="text/csv",
        width="stretch",
        help="Download this template to see the required column names and formats."
    )

uploaded_file = None
if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Online Retail XLSX or CSV", type=["xlsx", "csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Clustering Parameters")
k = st.sidebar.slider(
    "Number of clusters (k)", 
    CLUSTERING["k_min"], 
    CLUSTERING["k_max"], 
    CLUSTERING["default_k"],
    help="k is the number of groups you want to split your customers into."
)
m = st.sidebar.slider(
    "Fuzziness (m)", 
    CLUSTERING["m_min"], 
    CLUSTERING["m_max"], 
    CLUSTERING["default_m"],
    step=0.1,
    help="Higher values allow more overlap between clusters. 2.0 is the standard."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Feature Selection")
use_r = st.sidebar.checkbox("Recency", value=True, help="Days since the customer's last purchase.")
use_f = st.sidebar.checkbox("Frequency", value=True, help="Total number of unique orders made.")
use_m = st.sidebar.checkbox("Monetary", value=True, help="Total revenue generated from this customer.")

run_button = st.sidebar.button("🚀 Run Fuzzy Clustering", width="stretch")

st.sidebar.markdown("---")
st.sidebar.caption("⚖️ **Disclaimer**: This tool is for analytical purposes. Results should be validated by stakeholders before making financial commitments.")


# --- Main Logic ---
st.title("🎯 Customer Segmentation")
st.markdown("### Production-Grade Fuzzy C-Means Clustering on RFM Data")

with st.expander("📖 How to use this App", expanded=True):
    st.markdown("""
    Welcome! This tool uses **Fuzzy Logic** to segment your customers based on their buying behavior (**RFM Analysis**).
    
    ### 🛠️ Quick Start
    1. **Choose Data**: Select 'Use Sample Data' in the sidebar or upload your own CSV/XLSX.
    2. **Set Clusters**: Use the slider to pick how many groups (k) you want to identify.
    3. **Run Analysis**: Click the **🚀 Run** button in the sidebar.
    
    ### 🔍 What do the tabs show?
    - **📊 Data & RFM**: Your raw transactions converted into Recency, Frequency, and Monetary scores.
    - **🔬 Clustering Results**: The main grouping of your customers.
    - **🗺️ Membership Analysis**: Discover 'Ambiguous' customers who sit between two segments.
    - **👤 Predict Customer**: Enter data for a new customer to see which segment they join.
    - **💡 Business Insights**: Revenue-at-risk and specific marketing recommendations.
    - **⚖️ Model Comparison**: Comparison between this model and standard K-Means.
    """)


# Load Data
df_raw = None
if data_source == "Use Sample Data":
    df_raw = get_sample_data()
elif uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df_raw = pd.read_excel(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)
            
        # Fix PyArrow/Streamlit serialization issues for mixed-type columns
        for col in ["StockCode", "Description"]:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype(str)
                
        if not validate_columns(df_raw):
            df_raw = None
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👋 Welcome! Please upload a dataset or select 'Use Sample Data' in the sidebar to begin.")
    st.stop()

if df_raw is not None:
    # Preprocessing
    with st.spinner("Cleaning data and computing RFM..."):
        rfm_df = cached_preprocessing(df_raw)
    
    # Feature filtering
    features = []
    if use_r: features.append(RFM_CONFIG["recency_col"])
    if use_f: features.append(RFM_CONFIG["frequency_col"])
    if use_m: features.append(RFM_CONFIG["monetary_col"])
    
    if not features:
        st.warning("Please select at least one feature in the sidebar.")
        st.stop()
        
    rfm_subset = rfm_df[features]
    
    # Normalization
    normalized_df, scaler = normalize_rfm(rfm_subset, log_transform=RFM_CONFIG["log_transform"])
    
    # Run Clustering
    model_params = {
        "n_clusters": k,
        "fuzziness": m,
        "max_iter": CLUSTERING["max_iter"],
        "error": CLUSTERING["error_tolerance"],
        "random_state": CLUSTERING["random_state"]
    }
    
    try:
        result = cached_model_execution(normalized_df.values, model_params)
        cluster_meta = label_clusters(result.centers, features)
        cluster_names = [f"Cluster {i}: {m['label']}" for i, m in enumerate(cluster_meta)]
        
        # Ambiguous Customers
        ambiguous_df = get_ambiguous_customers(
            result.membership_matrix, 
            rfm_df.index, 
            AMBIGUITY_THRESHOLD
        )
        
        # Business Summary
        summary = generate_business_summary(
            rfm_df, result.labels, cluster_meta, len(ambiguous_df)
        )
    except Exception as e:
        st.error(f"Clustering failed: {e}")
        logger.exception("Clustering error")
        st.stop()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data & RFM", 
        "🔬 Clustering Results", 
        "🗺️ Membership Analysis", 
        "👤 Predict Customer", 
        "💡 Business Insights",
        "⚖️ Model Comparison"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Preview")
            st.dataframe(df_raw.head(), width="stretch")
        with col2:
            st.subheader("RFM Table Stats")
            st.dataframe(rfm_df.describe(), width="stretch")
        
        st.divider()
        st.subheader("RFM Distributions (Log-Transformed & Normalized)")
        st.plotly_chart(plot_rfm_distributions(normalized_df), use_container_width=True)

    with tab2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Partition Coefficient", f"{result.partition_coefficient:.3f}")
        m2.metric("Silhouette Score", f"{result.silhouette_score:.3f}")
        m3.metric("Total Clusters", k)
        m4.metric("Ambiguous Customers", len(ambiguous_df))
        
        col_x, col_y = st.columns(2)
        feat_x = col_x.selectbox("X Axis", features, index=0)
        feat_y = col_y.selectbox("Y Axis", features, index=min(1, len(features)-1))
        
        st.plotly_chart(
            plot_scatter_clusters(normalized_df, result.labels, result.membership_matrix, feat_x, feat_y),
            use_container_width=True
        )
        st.plotly_chart(plot_cluster_bar(result.labels, cluster_names), use_container_width=True)

    with tab3:
        st.plotly_chart(
            plot_membership_heatmap(result.membership_matrix, rfm_df.index, cluster_names),
            use_container_width=True
        )
        st.plotly_chart(
            plot_membership_distribution(result.membership_matrix, cluster_names),
            use_container_width=True
        )
        st.subheader("Ambiguous Customers (High Uncertainty)")
        st.markdown(f"Customers with max membership < {AMBIGUITY_THRESHOLD}")
        st.dataframe(ambiguous_df, width="stretch")

    with tab4:
        st.subheader("Predict Segment for New Customer")
        c1, c2, c3 = st.columns(3)
        in_r = c1.number_input("Recency (Days)", value=30.0)
        in_f = c2.number_input("Frequency (Count)", value=5.0)
        in_m = c3.number_input("Monetary (Spend)", value=500.0)
        
        if st.button("🔮 Predict", width="stretch"):
            # Prep input
            input_data = pd.DataFrame([[in_r, in_f, in_m]], columns=features)
            if RFM_CONFIG["log_transform"]:
                input_data = np.log1p(input_data)
            
            # Use the fitted scaler from training
            input_scaled = scaler.transform(input_data)
            
            # Predict
            u_new = predict_new_customer(input_scaled, result.centers, m)
            
            # Display result
            pred_idx = np.argmax(u_new)
            pred_meta = cluster_meta[pred_idx]
            
            st.success(f"Customer assigned to **{pred_meta['label']}** {pred_meta['emoji']}")
            
            # Probability bar chart
            prob_df = pd.DataFrame({
                "Cluster": cluster_names,
                "Probability": u_new
            })
            fig_new = px.bar(prob_df, x="Probability", y="Cluster", orientation='h', 
                             title="Membership Probabilities", text_auto=".2f")
            st.plotly_chart(fig_new, use_container_width=True)

    with tab5:
        st.subheader("Strategic Business Insights")
        
        i1, i2 = st.columns(2)
        i1.metric("Revenue at Risk 🚩", format_currency(summary["revenue_at_risk"]), delta_color="inverse")
        i2.metric("Champion Revenue 🏆", format_currency(summary["champion_revenue"]))
        
        st.divider()
        st.subheader("Segment Analysis & Recommendations")
        
        for i, meta in enumerate(cluster_meta):
            stats = summary["per_cluster_stats"][i]
            with st.expander(f"{meta['emoji']} {meta['label']} ({stats['count']} customers)"):
                st.write(f"**Average Recency:** {format_number(stats['avg_recency'], 1)} days")
                st.write(f"**Total Revenue Contribution:** {format_currency(stats['revenue'])}")
                st.info(f"💡 **Recommendation:** {meta['recommendations']}")
        
        st.divider()
        st.subheader("Targeted Campaign Suggestions")
        if summary["ambiguous_count"] > 0:
            st.warning(f"You have {summary['ambiguous_count']} customers in the 'Ambiguous' zone. "
                       "Avoid aggressive targeting; instead, use a 'Soft Discovery' campaign to learn more about their preferences.")
        else:
            st.success("Your customer base is well-segmented with low ambiguity!")

    with tab6:
        st.subheader("Fuzzy C-Means vs K-Means Comparison")
        st.markdown("""
        This section compares **Soft Clustering (Fuzzy C-Means)** with **Hard Clustering (K-Means)**. 
        In a portfolio, this demonstrates your understanding of model trade-offs.
        """)
        
        # Run K-Means for comparison
        with st.spinner("Running K-Means for comparison..."):
            km_result = cached_kmeans_execution(normalized_df.values, k)
            agreement = calculate_agreement(result.labels, km_result.labels)

        # Metrics Comparison
        c1, c2, c3 = st.columns(3)
        c1.metric("FCM Silhouette", f"{result.silhouette_score:.3f}")
        c2.metric("K-Means Silhouette", f"{km_result.silhouette_score:.3f}")
        c3.metric("Adjusted Rand Index", f"{agreement:.3f}", help="Measures similarity between two clusterings (1.0 is perfect agreement)")

        # Visual Comparison
        col_x2, col_y2 = st.columns(2)
        feat_x2 = col_x2.selectbox("X Axis ", features, index=0, key="comp_x")
        feat_y2 = col_y2.selectbox("Y Axis ", features, index=min(1, len(features)-1), key="comp_y")
        
        st.plotly_chart(
            plot_model_comparison(normalized_df, result.labels, km_result.labels, feat_x2, feat_y2),
            use_container_width=True
        )

        # Portfolio Insights
        st.info("### 🎓 Portfolio Note: Why Fuzzy C-Means?")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("""
            **Standard K-Means (Hard)**
            - Every customer belongs to exactly **one** cluster.
            - Great for simple segmentation.
            - Fails to capture customers who are "on the fence" between two behaviors.
            """)
        with col_right:
            st.markdown("""
            **Fuzzy C-Means (Soft)**
            - Customers have a **membership degree** (0-1) for every cluster.
            - Identifies 'Ambiguous' customers (those with ~0.5 membership in two groups).
            - Provides a more nuanced view of customer transitions.
            """)

