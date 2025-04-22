#!/usr/bin/env python3

"""
NeuraShield Feature Engineering Module
This module provides advanced feature engineering for network traffic data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from scipy import stats

def create_statistical_features(df, group_cols, value_cols):
    """
    Create statistical features based on grouping columns
    
    Args:
        df: DataFrame with network data
        group_cols: Columns to group by (e.g., ['src_ip', 'dst_ip'])
        value_cols: Columns to compute statistics for (e.g., ['bytes', 'pkts'])
    
    Returns:
        DataFrame with additional statistical features
    """
    result_df = df.copy()
    
    for group_col in group_cols:
        for value_col in value_cols:
            # Skip if column doesn't exist
            if group_col not in df.columns or value_col not in df.columns:
                continue
                
            # Get group statistics
            group_stats = df.groupby(group_col)[value_col].agg(
                ['mean', 'std', 'min', 'max', 'count']
            ).reset_index()
            
            # Rename columns to avoid conflicts
            group_stats.columns = [
                group_col,
                f"{group_col}_{value_col}_mean", 
                f"{group_col}_{value_col}_std",
                f"{group_col}_{value_col}_min", 
                f"{group_col}_{value_col}_max",
                f"{group_col}_{value_col}_count"
            ]
            
            # Merge with original dataframe
            result_df = result_df.merge(group_stats, on=group_col, how='left')
            
            # Fill NaN values
            for col in group_stats.columns[1:]:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(0)
    
    return result_df

def create_interaction_features(df, feature_pairs):
    """
    Create interaction features between pairs of columns
    
    Args:
        df: DataFrame with network data
        feature_pairs: List of column pairs to create interactions for
    
    Returns:
        DataFrame with additional interaction features
    """
    result_df = df.copy()
    
    for col1, col2 in feature_pairs:
        # Skip if columns don't exist
        if col1 not in df.columns or col2 not in df.columns:
            continue
            
        # Create multiplicative interaction
        result_df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        
        # Create ratio (with error handling)
        result_df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)
        
        # Create difference
        result_df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
    
    return result_df

def create_time_window_features(df, time_col, value_cols, windows=[10, 30, 60]):
    """
    Create time window aggregation features
    
    Args:
        df: DataFrame with network data
        time_col: Column containing timestamps
        value_cols: Columns to compute time windows for
        windows: List of window sizes in seconds
    
    Returns:
        DataFrame with additional time window features
    """
    # Convert time column to datetime if it's not already
    if time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            # If conversion fails, return original dataframe
            return df
    
    # Sort by time
    if time_col in df.columns:
        df = df.sort_values(by=time_col)
    else:
        # If time column doesn't exist, return original dataframe
        return df
    
    result_df = df.copy()
    
    for window in windows:
        for value_col in value_cols:
            # Skip if column doesn't exist
            if value_col not in df.columns:
                continue
                
            # Create rolling window features
            result_df[f"{value_col}_rolling_{window}s_mean"] = df[value_col].rolling(
                window=f"{window}s", on=time_col, min_periods=1
            ).mean()
            
            result_df[f"{value_col}_rolling_{window}s_std"] = df[value_col].rolling(
                window=f"{window}s", on=time_col, min_periods=1
            ).std()
            
            result_df[f"{value_col}_rolling_{window}s_max"] = df[value_col].rolling(
                window=f"{window}s", on=time_col, min_periods=1
            ).max()
            
            result_df[f"{value_col}_rolling_{window}s_count"] = df[value_col].rolling(
                window=f"{window}s", on=time_col, min_periods=1
            ).count()
    
    # Fill NaN values
    for col in result_df.columns:
        if col.startswith(tuple([f"{value_col}_rolling" for value_col in value_cols])):
            result_df[col] = result_df[col].fillna(0)
    
    return result_df

def create_advanced_features(df):
    """
    Create advanced features for network traffic data
    
    Args:
        df: DataFrame with network data
    
    Returns:
        DataFrame with additional advanced features
    """
    result_df = df.copy()
    
    # Create entropy features for categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        # Calculate value frequencies
        value_counts = df[col].value_counts(normalize=True)
        # Calculate entropy for each value
        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
        # Add entropy as a feature
        result_df[f"{col}_entropy"] = entropy
    
    # Create skewness and kurtosis for numeric columns
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_columns:
        # Skip if column has all zeros or single value
        if df[col].nunique() <= 1:
            continue
            
        # Calculate skewness
        result_df[f"{col}_skew"] = stats.skew(df[col].fillna(0))
        
        # Calculate kurtosis
        result_df[f"{col}_kurt"] = stats.kurtosis(df[col].fillna(0))
    
    return result_df

def apply_feature_engineering(df, include_time_features=True, include_statistical=True, 
                             include_interactions=True, include_advanced=True):
    """
    Apply comprehensive feature engineering to network traffic data
    
    Args:
        df: DataFrame with network data
        include_time_features: Whether to include time window features
        include_statistical: Whether to include statistical features
        include_interactions: Whether to include interaction features
        include_advanced: Whether to include advanced features
    
    Returns:
        DataFrame with engineered features
    """
    result_df = df.copy()
    
    # Define key columns for different data types
    # These are typical columns in network traffic data - adjust based on your actual data
    id_columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    stat_columns = ['bytes', 'pkts', 'duration']
    time_column = 'timestamp'
    
    # Filter to only include columns that exist in the DataFrame
    id_columns = [col for col in id_columns if col in df.columns]
    stat_columns = [col for col in stat_columns if col in df.columns]
    time_feature_available = time_column in df.columns
    
    # Apply feature engineering steps based on available columns and flags
    if include_statistical and id_columns and stat_columns:
        result_df = create_statistical_features(
            result_df, 
            group_cols=id_columns, 
            value_cols=stat_columns
        )
    
    if include_interactions and stat_columns and len(stat_columns) >= 2:
        # Create pairs of all statistical columns
        pairs = [(stat_columns[i], stat_columns[j]) 
                for i in range(len(stat_columns)) 
                for j in range(i+1, len(stat_columns))]
        
        result_df = create_interaction_features(result_df, pairs)
    
    if include_time_features and time_feature_available and stat_columns:
        result_df = create_time_window_features(
            result_df,
            time_col=time_column,
            value_cols=stat_columns
        )
    
    if include_advanced:
        result_df = create_advanced_features(result_df)
    
    # Drop rows with infinite values
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NA values
    result_df = result_df.fillna(0)
    
    return result_df

def dimensionality_reduction(X, n_components=0.95, method='pca'):
    """
    Apply dimensionality reduction to features
    
    Args:
        X: Feature matrix
        n_components: Number of components or variance to preserve (0.95 = 95%)
        method: Method to use ('pca' or 'standard')
    
    Returns:
        Reduced feature matrix
    """
    if method == 'pca':
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # Print explained variance
        if isinstance(n_components, float):
            print(f"PCA reduced features from {X.shape[1]} to {X_reduced.shape[1]} "
                  f"while preserving {n_components*100:.1f}% of variance")
        else:
            explained_variance = sum(pca.explained_variance_ratio_) * 100
            print(f"PCA reduced features from {X.shape[1]} to {X_reduced.shape[1]} "
                  f"explaining {explained_variance:.1f}% of variance")
        
        return X_reduced
    else:
        # Just return standardized data
        scaler = StandardScaler()
        return scaler.fit_transform(X)

def unsw_feature_engineering(df):
    """
    Apply feature engineering specifically for UNSW-NB15 dataset
    
    Args:
        df: DataFrame with UNSW-NB15 data
    
    Returns:
        DataFrame with engineered features tailored for UNSW-NB15
    """
    result_df = df.copy()
    
    # Create feature interactions specific to UNSW dataset
    # Protocol-specific features
    if 'proto' in df.columns and 'service' in df.columns:
        # Create protocol-service combination feature
        result_df['proto_service'] = df['proto'].astype(str) + "_" + df['service'].astype(str)
    
    # Connection features
    if 'sttl' in df.columns and 'dttl' in df.columns:
        # TTL difference is significant for certain attacks
        result_df['ttl_diff'] = (df['sttl'] - df['dttl']).abs()
    
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        # Bytes asymmetry is significant for data exfiltration
        result_df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        result_df['bytes_diff'] = df['sbytes'] - df['dbytes']
    
    # State features
    if 'state' in df.columns:
        # One-hot encode connection state
        state_dummies = pd.get_dummies(df['state'], prefix='state')
        result_df = pd.concat([result_df, state_dummies], axis=1)
    
    # Rate features
    if 'sload' in df.columns and 'dload' in df.columns:
        # Load asymmetry is significant for DoS attacks
        result_df['load_ratio'] = df['sload'] / (df['dload'] + 1)
        result_df['load_diff'] = df['sload'] - df['dload']
    
    # Create combined features across categories
    if all(col in df.columns for col in ['ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm']):
        # Combine connection features for improved detection
        result_df['ct_combined'] = df['ct_srv_src'] * df['ct_state_ttl'] + df['ct_dst_ltm']
    
    # Non-linear transformations
    for col in ['sbytes', 'dbytes', 'sload', 'dload', 'rate']:
        if col in df.columns:
            # Log transformation for heavy-tailed features
            result_df[f'{col}_log'] = np.log1p(df[col])
    
    # Special handling for duration-based features
    if 'dur' in df.columns:
        # Create duration-based rate metrics
        for col in ['sbytes', 'dbytes', 'sttl', 'dttl']:
            if col in df.columns:
                # Rate per second metrics
                result_df[f'{col}_per_sec'] = df[col] / (df['dur'] + 0.001)
    
    return result_df 