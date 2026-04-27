# 🎯 Customer Segmentation — Fuzzy C-Means MVP

A production-grade Streamlit application for advanced customer segmentation using **RFM Analysis** and **Fuzzy C-Means Clustering**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-segmentation-fuzzy.streamlit.app/)

## 📖 Overview

This project transforms raw transactional data into actionable business strategy. Unlike traditional clustering (K-Means), which assigns customers to a single group, this application uses **Fuzzy Logic** to determine "membership degrees." This allows businesses to identify "ambiguous" customers who sit between segments, enabling more nuanced marketing campaigns.

## 🚀 Key Features

- **RFM Engine**: Automated computation of Recency, Frequency, and Monetary metrics.
- **Fuzzy C-Means Clustering**: "Soft" segmentation that captures the complexity of customer behavior.
- **Ambiguity Detection**: Identify customers who don't strongly belong to any cluster for specialized targeting.
- **Interactive Visualizations**: High-quality Plotly charts including cluster scatter plots, membership heatmaps, and distribution violin plots.
- **Predictive Segmenting**: Real-time cluster assignment for hypothetical new customers.
- **Model Comparison**: Side-by-side evaluation against standard K-Means to demonstrate the value of "soft" clustering.
- **Strategic Insights**: Automated segment labeling (e.g., Champions, At-Risk) with tailored business recommendations.

## 🛠️ Tech Stack

- **Python 3.11+**
- **Streamlit**: Web interface and orchestration.
- **scikit-fuzzy**: Core clustering algorithm.
- **Pandas & NumPy**: Data manipulation and numerical processing.
- **Plotly**: Interactive, publication-quality visualizations.
- **Scikit-Learn**: Data normalization and performance metrics.

## 📂 Project Structure

```text
customer_segmentation/
├── app.py                  # Streamlit entry point
├── config.py               # Constants, thresholds, and segment labels
├── requirements.txt        # Project dependencies
├── data/                   # Data storage (online_retail.xlsx)
└── src/
    ├── preprocessing.py    # Data cleaning & normalization
    ├── rfm.py              # RFM metric computation
    ├── fuzzy_model.py      # Clustering logic & Fuzzy Result container
    ├── visualization.py    # Plotly chart builders
    ├── insights.py         # Business logic & recommendation engine
    ├── comparisons.py      # K-Means vs FCM comparison logic
    └── utils.py            # Shared helpers (logging, formatting)
```

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd customer-segmentation
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Data Input**:
   - Upload your own `online_retail.xlsx` file.
   - Or check **"Use Sample Data"** in the sidebar to explore features immediately.

3. **Adjust Parameters**:
   - Use the sidebar to tune the **Number of Clusters (k)** and **Fuzziness (m)**.
   - Select specific features (Recency, Frequency, Monetary) to include in the model.

## 🧠 How it Works

### RFM Analysis
- **Recency**: Days since last purchase (lower is better).
- **Frequency**: Total number of unique orders (higher is better).
- **Monetary**: Total lifetime spend (higher is better).

### Fuzzy C-Means (FCM)
FCM allows for "Partial Membership." While K-Means might put a customer strictly in "At-Risk," FCM might show they are 60% "At-Risk" and 40% "Loyal." This distinction is critical for high-value retention strategies.

### Normalization
The model applies a `log1p` transformation to handle skewed retail data and a `StandardScaler` to ensure all features contribute equally to the distance calculations.

---
**Developed with ❤️ for Advanced Business Analytics.**
