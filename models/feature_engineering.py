#!/usr/bin/env python3

"""
NeuraShield Feature Engineering Module
This module provides functions for advanced feature engineering in threat detection
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations


def generate_statistical_features(X, window_sizes=[3, 5, 10]):
    """
    Generate statistical features from the input data
    Args:
        X: Input features (numpy array)
        window_sizes: List of window sizes for rolling statistics
    Returns:
        Array of statistical features
    """
    try:
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier manipulation
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Initialize list to store new features
        stat_features = []
        
        # Global statistics across all features
        # Mean, std, min, max of each row
        row_mean = X_df.mean(axis=1).values.reshape(-1, 1)
        row_std = X_df.std(axis=1).values.reshape(-1, 1)
        row_min = X_df.min(axis=1).values.reshape(-1, 1)
        row_max = X_df.max(axis=1).values.reshape(-1, 1)
        row_median = X_df.median(axis=1).values.reshape(-1, 1)
        
        # Range and percentiles
        row_range = row_max - row_min
        row_ptp = row_range.reshape(-1, 1)  # peak to peak
        row_skew = X_df.skew(axis=1).values.reshape(-1, 1)
        row_kurt = X_df.kurtosis(axis=1).values.reshape(-1, 1)
        
        # Add to feature list
        stat_features.extend([
            row_mean, row_std, row_min, row_max, row_median,
            row_ptp, row_skew, row_kurt
        ])
        
        # Calculate pair-wise ratios for significant features (if not too many)
        if X.shape[1] <= 20:  # Only do this for reasonably sized feature sets
            # Use most significant columns
            top_cols = list(range(min(10, X.shape[1])))
            for i, j in combinations(top_cols, 2):
                # Avoid division by zero by adding small epsilon
                ratio = X_df.iloc[:, i] / (X_df.iloc[:, j] + 1e-10)
                stat_features.append(ratio.values.reshape(-1, 1))
        
        # Concatenate all features
        if stat_features:
            X_stats = np.hstack(stat_features)
            print(f"Generated {X_stats.shape[1]} statistical features")
            return X_stats
        else:
            return None
    except Exception as e:
        print(f"Error generating statistical features: {e}")
        return None


def generate_interaction_features(X, top_n=5):
    """
    Generate interaction features between top feature pairs
    Args:
        X: Input features (numpy array)
        top_n: Number of top features to consider for interactions
    Returns:
        Array of interaction features
    """
    try:
        if X.shape[1] <= 1:
            return None
            
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier manipulation
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        n_features = min(top_n, X_df.shape[1])
        top_cols = list(range(n_features))
        
        interaction_features = []
        
        # Generate multiplicative interactions
        for i, j in combinations(top_cols, 2):
            # Multiplication
            mult = (X_df.iloc[:, i] * X_df.iloc[:, j]).values.reshape(-1, 1)
            interaction_features.append(mult)
            
            # Addition
            add = (X_df.iloc[:, i] + X_df.iloc[:, j]).values.reshape(-1, 1)
            interaction_features.append(add)
            
            # Subtraction
            sub = (X_df.iloc[:, i] - X_df.iloc[:, j]).values.reshape(-1, 1)
            interaction_features.append(sub)
            
            # Division (with protection against division by zero)
            div = (X_df.iloc[:, i] / (X_df.iloc[:, j] + 1e-10)).values.reshape(-1, 1)
            interaction_features.append(div)
        
        # Concatenate all features
        if interaction_features:
            X_interactions = np.hstack(interaction_features)
            print(f"Generated {X_interactions.shape[1]} interaction features")
            return X_interactions
        else:
            return None
    except Exception as e:
        print(f"Error generating interaction features: {e}")
        return None


def generate_advanced_features(X, feature_names=None):
    """
    Generate domain-specific advanced features based on feature names
    Args:
        X: Input features (numpy array)
        feature_names: Names of features (list)
    Returns:
        Array of advanced features
    """
    try:
        if feature_names is None or len(feature_names) != X.shape[1]:
            return None
            
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier manipulation
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
        
        advanced_features = []
        
        # Look for network-related feature patterns
        network_features = [col for col in feature_names if any(kw in col.lower() for kw in [
            'packet', 'flow', 'byte', 'port', 'ip', 'conn', 'tcp', 'udp', 'http', 'dns'
        ])]
        
        # Look for time-related feature patterns
        time_features = [col for col in feature_names if any(kw in col.lower() for kw in [
            'time', 'duration', 'interval', 'delay', 'period', 'second', 'minute'
        ])]
        
        # Look for count-related feature patterns
        count_features = [col for col in feature_names if any(kw in col.lower() for kw in [
            'count', 'num', 'number', 'freq', 'frequency'
        ])]
        
        # Network feature engineering
        if len(network_features) >= 2:
            # Extract relevant columns
            network_df = X_df[network_features]
            
            # Calculate packet rate if we have packet counts and duration
            packet_cols = [col for col in network_features if 'packet' in col.lower()]
            duration_cols = [col for col in time_features if 'duration' in col.lower() or 'time' in col.lower()]
            
            if packet_cols and duration_cols:
                for p_col in packet_cols:
                    for d_col in duration_cols:
                        # Packet rate: packets per unit time
                        packet_rate = (network_df[p_col] / (X_df[d_col] + 1e-10)).values.reshape(-1, 1)
                        advanced_features.append(packet_rate)
            
            # Calculate byte rate if we have byte counts and duration
            byte_cols = [col for col in network_features if 'byte' in col.lower()]
            if byte_cols and duration_cols:
                for b_col in byte_cols:
                    for d_col in duration_cols:
                        # Byte rate: bytes per unit time
                        byte_rate = (network_df[b_col] / (X_df[d_col] + 1e-10)).values.reshape(-1, 1)
                        advanced_features.append(byte_rate)
            
            # Average packet size if we have both packets and bytes
            if packet_cols and byte_cols:
                for b_col in byte_cols:
                    for p_col in packet_cols:
                        # Average packet size: bytes per packet
                        avg_packet_size = (network_df[b_col] / (network_df[p_col] + 1e-10)).values.reshape(-1, 1)
                        advanced_features.append(avg_packet_size)
        
        # Time feature engineering
        if len(time_features) >= 2:
            # Extract relevant columns
            time_df = X_df[time_features]
            
            # Calculate time-based ratios
            for i, t1 in enumerate(time_features):
                for t2 in time_features[i+1:]:
                    # Time ratios
                    time_ratio = (time_df[t1] / (time_df[t2] + 1e-10)).values.reshape(-1, 1)
                    advanced_features.append(time_ratio)
        
        # Count feature engineering
        if len(count_features) >= 2:
            # Extract relevant columns
            count_df = X_df[count_features]
            
            # Calculate count-based ratios
            for i, c1 in enumerate(count_features):
                for c2 in count_features[i+1:]:
                    # Count ratios
                    count_ratio = (count_df[c1] / (count_df[c2] + 1e-10)).values.reshape(-1, 1)
                    advanced_features.append(count_ratio)
        
        # Entropy-based features for distributions
        # If we have multiple related features, calculate entropy of their distribution
        feature_groups = [network_features, time_features, count_features]
        for group in feature_groups:
            if len(group) >= 3:  # Need at least 3 features to make this meaningful
                group_values = X_df[group].values
                # Normalize each row to get distribution
                row_sums = np.sum(np.abs(group_values), axis=1) + 1e-10
                normalized = group_values / row_sums.reshape(-1, 1)
                # Calculate entropy of distribution
                entropy = np.sum(-normalized * np.log2(normalized + 1e-10), axis=1).reshape(-1, 1)
                advanced_features.append(entropy)
        
        # Concatenate all features
        if advanced_features:
            X_advanced = np.hstack(advanced_features)
            print(f"Generated {X_advanced.shape[1]} advanced domain-specific features")
            return X_advanced
        else:
            return None
    except Exception as e:
        print(f"Error generating advanced features: {e}")
        return None


def select_important_features(X, y, threshold=0.05, method='mutual_info'):
    """
    Select important features based on their relationship with the target
    Args:
        X: Input features (numpy array)
        y: Target variable (numpy array)
        threshold: Importance threshold for feature selection
        method: Feature selection method ('mutual_info' or 'correlation')
    Returns:
        Boolean mask of selected features
    """
    try:
        if method == 'mutual_info':
            # Calculate mutual information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            # Normalize scores
            mi_scores = mi_scores / np.max(mi_scores)
            # Select features above threshold
            selected = mi_scores > threshold
            
        elif method == 'correlation':
            # Calculate correlation for each feature
            corr_scores = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
            # Select features above threshold
            selected = corr_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Selected {np.sum(selected)} features out of {len(selected)}")
        return selected
    except Exception as e:
        print(f"Error in feature selection: {e}")
        return np.ones(X.shape[1], dtype=bool)  # Default: select all features


def remove_redundant_features(X, threshold=0.95):
    """
    Remove highly correlated features to reduce redundancy
    Args:
        X: Input features (numpy array)
        threshold: Correlation threshold for redundancy
    Returns:
        Boolean mask of selected features
    """
    try:
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Calculate correlation matrix
        corr_matrix = X_df.corr().abs()
        
        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Create boolean mask of features to keep
        keep = ~X_df.columns.isin(to_drop)
        
        print(f"Removed {len(to_drop)} redundant features, keeping {sum(keep)}")
        return keep
    except Exception as e:
        print(f"Error removing redundant features: {e}")
        return np.ones(X.shape[1], dtype=bool)  # Default: keep all features


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module - Run this as part of the fine-tuning process") 