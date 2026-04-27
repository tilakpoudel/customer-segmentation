# src/preprocessing.py
"""Data loading, cleaning, and normalization logic."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import setup_logging

logger = setup_logging()

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load xlsx, return raw dataframe.
    
    Args:
        filepath (str): Path to the Excel file.
        
    Returns:
        pd.DataFrame: Loaded raw data.
    """
    try:
        logger.info(f"Loading data from {filepath}")
        return pd.read_excel(filepath)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Online Retail dataset.
    
    - Drop null CustomerID
    - Remove negative Quantity and UnitPrice
    - Remove duplicate InvoiceNo+StockCode rows
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned copy of the dataframe.
    """
    df_clean = df.copy()
    
    # Drop rows with missing CustomerID
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=["CustomerID"])
    
    # Ensure CustomerID is integer (it often loads as float)
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
    
    # Ensure StockCode and Description are strings for Arrow compatibility
    for col in ["StockCode", "Description"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
    
    # Remove negative or zero Quantity and UnitPrice
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    
    # Remove duplicates for same InvoiceNo and StockCode
    df_clean = df_clean.drop_duplicates(subset=["InvoiceNo", "StockCode"])
    
    final_count = len(df_clean)
    logger.info(f"Cleaned data: {initial_count} -> {final_count} rows")
    
    return df_clean

def normalize_rfm(rfm_df: pd.DataFrame, log_transform: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize RFM features.
    
    - Optionally log1p transform each column
    - StandardScaler normalize
    
    Args:
        rfm_df (pd.DataFrame): Dataframe with Recency, Frequency, Monetary columns.
        log_transform (bool): Whether to apply log transformation.
        
    Returns:
        tuple[pd.DataFrame, StandardScaler]: (Scaled dataframe, fitted scaler).
    """
    df_scaled = rfm_df.copy()
    
    # 1. Log Transformation
    if log_transform:
        # log1p handles potential zero values gracefully
        df_scaled = np.log1p(df_scaled)
    
    # 2. Standardization
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_scaled)
    
    normalized_df = pd.DataFrame(
        scaled_values, 
        index=rfm_df.index, 
        columns=rfm_df.columns
    )
    return normalized_df, scaler
