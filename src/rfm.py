# src/rfm.py
"""RFM (Recency, Frequency, Monetary) computation logic."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.utils import setup_logging

logger = setup_logging()

def compute_rfm(
    df: pd.DataFrame, 
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Compute RFM metrics from transactional data.
    
    - Recency: Days since last purchase (lower = better)
    - Frequency: Number of unique invoices
    - Monetary: Total spend
    
    Args:
        df (pd.DataFrame): Cleaned transactional data.
        reference_date (Optional[datetime]): Date to calculate recency from. 
            Defaults to max(InvoiceDate) + 1 day.
            
    Returns:
        pd.DataFrame: Dataframe indexed by CustomerID with Recency, Frequency, Monetary.
    """
    logger.info("Computing RFM metrics")
    
    # Ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    # Calculate Total Price
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + timedelta(days=1)
    
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })
    
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    
    logger.info(f"Computed RFM for {len(rfm)} customers")
    
    return rfm
