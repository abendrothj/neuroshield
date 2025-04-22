#!/usr/bin/env python3

"""
NeuraShield Inference Module
This module provides classes for loading the trained model and making predictions.
"""

import os
import sys
import time
import logging
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Union, Tuple, Any, Optional
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

# Set TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PlattScaler:
    """Class for calibrating probabilities using Platt scaling."""
    
    def __init__(self, A=1.0, B=0.0):
        """
        Initialize the Platt scaler.
        
        Args:
            A: Platt scaling parameter A (scaling)
            B: Platt scaling parameter B (shift)
        """
        self.A = A
        self.B = B
        
    def calibrate(self, prob):
        """
        Calibrate a probability using Platt scaling.
        
        Args:
            prob: Raw probability
            
        Returns:
            Calibrated probability
        """
        # Convert probability to logit
        try:
            if prob >= 1.0:
                prob = 0.999
            elif prob <= 0.0:
                prob = 0.001
                
            logit = np.log(prob / (1 - prob))
            
            # Apply Platt scaling
            scaled_logit = self.A * logit + self.B
            
            # Convert back to probability
            calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
            
            return float(calibrated_prob)
        except Exception as e:
            logging.warning(f"Error in Platt scaling: {str(e)}")
            return prob

class ConfidenceCalibrator:
    """Class for calibrating confidence scores from model predictions."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize the calibrator.
        
        Args:
            temperature: Temperature parameter for scaling logits
        """
        self.temperature = temperature
        self.epsilon = 1e-10  # Small value to prevent division by zero
        self.min_confidence = 0.2  # Increased minimum confidence
        self.max_confidence = 0.9  # Reduced maximum confidence
        self.conservative_factor = 0.8  # Factor to make predictions more conservative
        
        # Platt scaling for better probability calibration
        self.platt_scaler = PlattScaler(A=0.5, B=-0.5)  # Parameters tuned for more conservative output
    
    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a confidence score.
        
        Args:
            confidence: Raw confidence score from model
            
        Returns:
            Calibrated confidence score
        """
        # Handle edge cases with epsilon to prevent division by zero
        if confidence >= 1.0:
            confidence = 1.0 - self.epsilon
        elif confidence <= 0.0:
            confidence = self.epsilon
            
        # Apply Platt scaling first
        calibrated_confidence = self.platt_scaler.calibrate(confidence)
        
        # Then apply temperature scaling with numerical stability
        try:
            # Apply sigmoid to get more realistic confidence scores
            logit = np.log(calibrated_confidence / (1 - calibrated_confidence))
            calibrated_logit = logit / self.temperature
            
            # Apply conservative factor
            calibrated_logit = calibrated_logit * self.conservative_factor
            
            calibrated_confidence = 1 / (1 + np.exp(-calibrated_logit))
            
            # Apply confidence bounds
            calibrated_confidence = max(self.min_confidence, 
                                     min(self.max_confidence, 
                                         calibrated_confidence))
            
            return float(calibrated_confidence)
        except Exception as e:
            logging.warning(f"Error in confidence calibration: {str(e)}")
            # Return a conservative confidence value if calibration fails
            return self.min_confidence

class FeatureEngineer:
    """Class for enhancing features with derived metrics."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        # Define feature importance based on training analysis
        self.feature_importance = {
            "feature_0": 0.60,  # Flow duration (reduced importance)
            "feature_1": 0.55,  # Total bytes (reduced importance)
            "feature_2": 0.50,  # Packet count (reduced importance)
            "feature_3": 0.75,  # Byte rate (reduced from 0.90)
            "feature_4": 0.45,  # TTL (reduced importance)
            "feature_5": 0.30,  # Window size (reduced importance)
            "feature_6": 0.50,  # Packet size (reduced importance)
            "feature_7": 0.70   # Inter-arrival time (reduced from 0.85)
        }
        
        # Define normalization parameters for derived features
        self.normalization_params = {
            "derived_bytes_per_packet": {"max": 2000, "min": 0},  # Increased max
            "derived_packets_per_second": {"max": 2000, "min": 0},  # Increased max
            "derived_burst_ratio": {"max": 30000, "min": 0},  # Increased max
            "derived_traffic_intensity": {"max": 25, "min": 0},  # Increased max
            "derived_ttl_anomaly": {"max": 1, "min": 0}
        }
    
    def _normalize_derived_feature(self, value: float, feature_name: str) -> float:
        """Normalize a derived feature value to [0,1] range."""
        params = self.normalization_params.get(feature_name)
        if not params:
            return value
            
        # Apply min-max normalization
        normalized = (value - params["min"]) / (params["max"] - params["min"])
        # Clip to [0,1] range
        return max(0.0, min(1.0, normalized))
    
    def enhance_features(self, features):
        """
        Add derived features to improve detection.
        
        Args:
            features: Original feature dictionary or DataFrame
            
        Returns:
            Enhanced feature dictionary or DataFrame
        """
        # Handle DataFrame input
        if isinstance(features, pd.DataFrame):
            enhanced = features.copy()
            
            # Add derived features based on domain knowledge
            if set(["feature_1", "feature_2"]).issubset(enhanced.columns):
                # Bytes per packet (useful for detecting payload anomalies)
                enhanced["derived_bytes_per_packet"] = enhanced["feature_1"] / enhanced["feature_2"].replace(0, 1)
                enhanced["derived_bytes_per_packet"] = enhanced["derived_bytes_per_packet"].apply(
                    lambda x: self._normalize_derived_feature(x, "derived_bytes_per_packet")
                )
                
                # Add packet size anomaly detection
                enhanced["derived_packet_size_anomaly"] = abs(enhanced["derived_bytes_per_packet"] - 0.5) * 2
            
            if set(["feature_0", "feature_2"]).issubset(enhanced.columns):
                # Packets per second (useful for detecting flooding attacks)
                enhanced["derived_packets_per_second"] = enhanced["feature_2"] / enhanced["feature_0"].replace(0, 1)
                enhanced["derived_packets_per_second"] = enhanced["derived_packets_per_second"].apply(
                    lambda x: self._normalize_derived_feature(x, "derived_packets_per_second")
                )
                
                # Add traffic rate anomaly detection
                enhanced["derived_traffic_rate_anomaly"] = abs(enhanced["derived_packets_per_second"] - 0.5) * 2
            
            if set(["feature_3", "feature_7"]).issubset(enhanced.columns):
                # Byte rate / inter-arrival time ratio (useful for detecting bursts)
                enhanced["derived_burst_ratio"] = enhanced["feature_3"] / enhanced["feature_7"].replace(0, 1)
                enhanced["derived_burst_ratio"] = enhanced["derived_burst_ratio"].apply(
                    lambda x: self._normalize_derived_feature(x, "derived_burst_ratio")
                )
                
                # Add burst anomaly detection
                enhanced["derived_burst_anomaly"] = abs(enhanced["derived_burst_ratio"] - 0.5) * 2
            
            # Add feature relationship metrics
            if set(["feature_2", "feature_3"]).issubset(enhanced.columns):
                # Traffic intensity metric (combines packet count and byte rate)
                enhanced["derived_traffic_intensity"] = np.log1p(enhanced["feature_2"] * enhanced["feature_3"])
                enhanced["derived_traffic_intensity"] = enhanced["derived_traffic_intensity"].apply(
                    lambda x: self._normalize_derived_feature(x, "derived_traffic_intensity")
                )
                
                # Add traffic pattern anomaly detection
                enhanced["derived_traffic_pattern_anomaly"] = abs(enhanced["derived_traffic_intensity"] - 0.5) * 2
            
            # Detect anomalous TTL values (very low or non-standard)
            if "feature_4" in enhanced.columns:
                # Calculate distance from common TTL values (32, 64, 128, 255)
                ttl_distances = pd.DataFrame({
                    "d32": abs(enhanced["feature_4"] - 32),
                    "d64": abs(enhanced["feature_4"] - 64),
                    "d128": abs(enhanced["feature_4"] - 128),
                    "d255": abs(enhanced["feature_4"] - 255)
                })
                enhanced["derived_ttl_anomaly"] = ttl_distances.min(axis=1) / 255.0
                enhanced["derived_ttl_anomaly"] = enhanced["derived_ttl_anomaly"].apply(
                    lambda x: self._normalize_derived_feature(x, "derived_ttl_anomaly")
                )
            
            # Calculate overall anomaly score
            anomaly_columns = [col for col in enhanced.columns if col.endswith("_anomaly")]
            if anomaly_columns:
                enhanced["overall_anomaly_score"] = enhanced[anomaly_columns].mean(axis=1)
            
            return enhanced
            
        # Handle dictionary input (single sample)
        elif isinstance(features, dict):
            enhanced = features.copy()
            
            # Add derived features based on domain knowledge
            if "feature_1" in features and "feature_2" in features:
                # Bytes per packet (useful for detecting payload anomalies)
                enhanced["derived_bytes_per_packet"] = self._normalize_derived_feature(
                    features["feature_1"] / max(1, features["feature_2"]),
                    "derived_bytes_per_packet"
                )
                enhanced["derived_packet_size_anomaly"] = abs(enhanced["derived_bytes_per_packet"] - 0.5) * 2
            
            if "feature_0" in features and "feature_2" in features:
                # Packets per second (useful for detecting flooding attacks)
                enhanced["derived_packets_per_second"] = self._normalize_derived_feature(
                    features["feature_2"] / max(1, features["feature_0"]),
                    "derived_packets_per_second"
                )
                enhanced["derived_traffic_rate_anomaly"] = abs(enhanced["derived_packets_per_second"] - 0.5) * 2
            
            if "feature_3" in features and "feature_7" in features:
                # Byte rate / inter-arrival time ratio (useful for detecting bursts)
                enhanced["derived_burst_ratio"] = self._normalize_derived_feature(
                    features["feature_3"] / max(1, features["feature_7"]),
                    "derived_burst_ratio"
                )
                enhanced["derived_burst_anomaly"] = abs(enhanced["derived_burst_ratio"] - 0.5) * 2
            
            # Add feature relationship metrics
            if "feature_2" in features and "feature_3" in features:
                # Traffic intensity metric (combines packet count and byte rate)
                enhanced["derived_traffic_intensity"] = self._normalize_derived_feature(
                    np.log1p(features["feature_2"] * features["feature_3"]),
                    "derived_traffic_intensity"
                )
                enhanced["derived_traffic_pattern_anomaly"] = abs(enhanced["derived_traffic_intensity"] - 0.5) * 2
            
            # Detect anomalous TTL values (very low or non-standard)
            if "feature_4" in features:
                ttl = features["feature_4"]
                # Calculate distance from common TTL values (32, 64, 128, 255)
                ttl_standard_distance = min(abs(ttl - 32), abs(ttl - 64), abs(ttl - 128), abs(ttl - 255))
                enhanced["derived_ttl_anomaly"] = self._normalize_derived_feature(
                    ttl_standard_distance / 255.0,
                    "derived_ttl_anomaly"
                )
            
            # Calculate overall anomaly score
            anomaly_scores = [v for k, v in enhanced.items() if k.endswith("_anomaly")]
            if anomaly_scores:
                enhanced["overall_anomaly_score"] = np.mean(anomaly_scores)
            
            return enhanced
            
        else:
            # If neither DataFrame nor dict, return unchanged
            logging.warning(f"Unsupported input type for feature enhancement: {type(features)}")
            return features

class ProbabilityCalibrator:
    """Advanced probability calibration system using multiple techniques."""
    
    def __init__(self):
        """Initialize the probability calibrator with multiple calibration methods."""
        # Platt scaling with aggressive parameters to spread probabilities
        self.platt_aggressive = PlattScaler(A=0.3, B=-2.0)
        self.platt_moderate = PlattScaler(A=0.5, B=-1.0)
        self.platt_conservative = PlattScaler(A=0.7, B=-0.5)
        
        # Histogram binning parameters (10 bins with preset calibrated values)
        self.bin_edges = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
        self.bin_calibrated = [0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.95]
        
        # Isotonic regression approximation (monotonically increasing function)
        self.isotonic_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.isotonic_y = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 0.95]
    
    def _histogram_calibrate(self, prob):
        """Apply histogram binning calibration."""
        # Find the bin index
        bin_idx = np.digitize(prob, self.bin_edges) - 1
        # Handle edge case
        if bin_idx >= len(self.bin_calibrated):
            bin_idx = len(self.bin_calibrated) - 1
        elif bin_idx < 0:
            bin_idx = 0
        return self.bin_calibrated[bin_idx]
    
    def _isotonic_calibrate(self, prob):
        """Apply isotonic regression calibration (approximated)."""
        # Find the position in isotonic_x
        for i in range(len(self.isotonic_x) - 1):
            if self.isotonic_x[i] <= prob <= self.isotonic_x[i+1]:
                # Linear interpolation
                range_x = self.isotonic_x[i+1] - self.isotonic_x[i]
                range_y = self.isotonic_y[i+1] - self.isotonic_y[i]
                pos = (prob - self.isotonic_x[i]) / range_x
                return self.isotonic_y[i] + pos * range_y
        
        # Default return if outside range
        return 0.5
    
    def _beta_calibrate(self, prob, alpha=2.0, beta=5.0):
        """Apply Beta distribution transformation for smoothing."""
        from scipy.stats import beta
        # Convert to position in the beta CDF
        if prob <= 0.0:
            return 0.01
        elif prob >= 1.0:
            return 0.99
        else:
            # This creates a smooth S-curve that pushes values toward the center
            return float(beta.cdf(prob, alpha, beta))
    
    def calibrate(self, probability, anomaly_score=0.5, method='ensemble'):
        """
        Calibrate a probability using multiple methods.
        
        Args:
            probability: Raw probability from model
            anomaly_score: Anomaly score (0-1) to influence calibration
            method: Calibration method ('platt', 'isotonic', 'histogram', 'beta', 'ensemble')
            
        Returns:
            Calibrated probability
        """
        try:
            # Sanitize input
            if probability >= 1.0:
                probability = 0.999
            elif probability <= 0.0:
                probability = 0.001
            
            # Apply appropriate calibration based on method
            if method == 'platt':
                # Select Platt scaler based on anomaly score
                if anomaly_score > 0.7:
                    return self.platt_conservative.calibrate(probability)
                elif anomaly_score < 0.3:
                    return self.platt_aggressive.calibrate(probability)
                else:
                    return self.platt_moderate.calibrate(probability)
            
            elif method == 'isotonic':
                return self._isotonic_calibrate(probability)
            
            elif method == 'histogram':
                return self._histogram_calibrate(probability)
            
            elif method == 'beta':
                return self._beta_calibrate(probability)
            
            elif method == 'ensemble':
                # Weighted ensemble of multiple methods
                platt_prob = self.platt_moderate.calibrate(probability)
                iso_prob = self._isotonic_calibrate(probability)
                hist_prob = self._histogram_calibrate(probability)
                beta_prob = self._beta_calibrate(probability)
                
                # Weight based on anomaly score
                if anomaly_score > 0.7:
                    # High anomaly - trust histogram and isotonic more
                    weights = [0.1, 0.4, 0.4, 0.1]  # platt, iso, hist, beta
                elif anomaly_score < 0.3:
                    # Low anomaly - trust platt and beta more
                    weights = [0.4, 0.1, 0.1, 0.4]
                else:
                    # Balanced weights
                    weights = [0.25, 0.25, 0.25, 0.25]
                
                # Weighted average
                calibrated = (
                    weights[0] * platt_prob +
                    weights[1] * iso_prob +
                    weights[2] * hist_prob +
                    weights[3] * beta_prob
                )
                
                return float(calibrated)
            
            else:
                # Default to moderate Platt scaling
                return self.platt_moderate.calibrate(probability)
                
        except Exception as e:
            logging.warning(f"Error in probability calibration: {str(e)}")
            # Return a conservative value
            return 0.3

class ThresholdDecisionMaker:
    """Advanced threshold decision system with multiple criteria."""
    
    def __init__(self):
        """Initialize the threshold decision maker."""
        # Base thresholds for different scenarios
        self.base_thresholds = {
            "default": 0.15,
            "high_traffic": 0.20,
            "low_traffic": 0.10,
            "high_anomaly": 0.12,
            "low_anomaly": 0.18,
            "sensitive": 0.08,
            "conservative": 0.25
        }
        
        # Feature importance weights for decision
        self.feature_weights = {
            "probability": 0.3,
            "anomaly_score": 0.3,
            "confidence": 0.2,
            "feature_variance": 0.1,
            "derived_features": 0.1
        }
    
    def _get_feature_variance(self, features):
        """Calculate variance across features as a measure of uniformity."""
        if isinstance(features, np.ndarray):
            return np.var(features)
        return 0.0
    
    def _get_attack_pattern_score(self, features, anomaly_score):
        """
        Calculate a score based on known attack patterns.
        Higher score means more likely to be an attack.
        """
        # If high anomaly, already suspicious
        if anomaly_score > 0.7:
            return 0.7
            
        # Calculate ratio-based metrics
        try:
            # Look for DDoS patterns - high packet counts, low bytes per packet
            if features[2] > 0.7 and features[1] / max(1, features[2]) < 0.3:
                return 0.8
                
            # Look for port scanning - many packets, short duration
            if features[2] > 0.6 and features[0] < 0.2:
                return 0.75
                
            # Look for data exfiltration - high byte counts, few packets
            if features[1] > 0.8 and features[2] < 0.3:
                return 0.7
        except:
            pass
            
        # Default - no specific pattern detected
        return 0.5
    
    def make_decision(self, probability, anomaly_score, confidence, features=None, derived_features=None):
        """
        Determine if a sample is an attack using multiple criteria.
        
        Args:
            probability: Calibrated probability from model
            anomaly_score: Anomaly score (0-1)
            confidence: Model confidence
            features: Raw feature values (optional)
            derived_features: Derived feature values (optional)
            
        Returns:
            is_attack: Boolean decision
            effective_threshold: The threshold used
            explanation: Dict with explanation of decision factors
        """
        # Start with default threshold
        threshold = self.base_thresholds["default"]
        
        # Initialize decision factors
        decision_factors = {}
        
        # 1. Check if we have specific attack patterns
        pattern_score = 0.5
        if features is not None:
            # Calculate variance for uniformity check
            variance = self._get_feature_variance(features)
            decision_factors["feature_variance"] = variance
            
            # Check for known attack patterns
            pattern_score = self._get_attack_pattern_score(features, anomaly_score)
            decision_factors["attack_pattern_score"] = pattern_score
            
            # Adjust threshold based on variance
            if variance > 0.5:  # High variance - less uniform traffic
                threshold = (threshold + self.base_thresholds["high_traffic"]) / 2
            elif variance < 0.1:  # Low variance - more uniform traffic
                threshold = (threshold + self.base_thresholds["low_traffic"]) / 2
        
        # 2. Adjust based on anomaly score
        if anomaly_score > 0.7:  # High anomaly
            decision_factors["high_anomaly"] = True
            threshold = (threshold + self.base_thresholds["high_anomaly"]) / 2
        elif anomaly_score < 0.3:  # Low anomaly
            decision_factors["low_anomaly"] = True
            threshold = (threshold + self.base_thresholds["low_anomaly"]) / 2
        
        # 3. Adjust based on confidence
        if confidence < 0.3:  # Low confidence
            decision_factors["low_confidence"] = True
            # Be more conservative with low confidence
            threshold = (threshold + self.base_thresholds["conservative"]) / 2
        elif confidence > 0.7:  # High confidence
            decision_factors["high_confidence"] = True
            # Can be more sensitive with high confidence
            threshold = (threshold + self.base_thresholds["sensitive"]) / 2
        
        # 4. Special case: strong attack pattern evidence
        if pattern_score > 0.75:
            decision_factors["strong_attack_pattern"] = True
            # Reduce threshold for clear attack patterns
            threshold = (threshold + self.base_thresholds["sensitive"]) / 2
            
        # 5. Special case: derived features show strong signal
        if derived_features is not None and any(v > 0.8 for v in derived_features):
            decision_factors["strong_derived_features"] = True
            # Reduce threshold for strong derived features
            threshold = (threshold + self.base_thresholds["sensitive"]) / 2
        
        # Ensure threshold is in reasonable range
        threshold = max(0.05, min(0.3, threshold))
        
        # Make final decision
        is_attack = probability >= threshold
        
        # Add threshold to factors
        decision_factors["effective_threshold"] = threshold
        decision_factors["probability"] = probability
        
        return is_attack, threshold, decision_factors

class NeuraShieldPredictor:
    """Class for making predictions using the trained NeuraShield model."""
    
    def __init__(
        self, 
        model_path: str = None, 
        scaler_path: str = None,
        feature_map_path: str = None,
        threshold: float = 0.15,  # Adjusted to 0.15 for better balance
        calibrate_confidence: bool = True,
        use_ensemble: bool = True,  # Default to using ensemble
        ensemble_paths: List[str] = None
    ):
        """
        Initialize the predictor with the trained model and scaler.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the scaler used to normalize features
            feature_map_path: Path to the feature mapping file
            threshold: Threshold for binary classification (default: 0.15)
            calibrate_confidence: Whether to calibrate confidence scores
            use_ensemble: Whether to use an ensemble of models
            ensemble_paths: Paths to additional models for ensemble
        """
        # Set default paths if not provided
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        # Set parameters with conservative thresholds
        self.base_threshold = threshold
        self.threshold_low = 0.12  # Lower bound for threshold
        self.threshold_high = 0.25  # Upper bound for threshold
        self.use_ensemble = use_ensemble
        self.calibrate_confidence = calibrate_confidence
        
        # Create feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Create probability calibrator with careful tuning
        self.probability_calibrator = ProbabilityCalibrator()
        
        # Create confidence calibrator with adjusted temperature
        self.confidence_calibrator = ConfidenceCalibrator(temperature=2.5)  # Higher temperature
        
        # Create threshold decision maker
        self.threshold_decision_maker = ThresholdDecisionMaker()
        
        # Locate the model if not specified
        if model_path is None:
            # Look in standard locations
            possible_paths = [
                # Check multi-dataset directory first (with improved designation)
                os.path.join(self.model_dir, "multi_dataset/chained_transfer_improved/best_model.keras"),
                os.path.join(self.model_dir, "multi_dataset/chained_transfer_improved/final_model.keras"),
                # Fall back to other possible locations
                os.path.join(self.model_dir, "multi_dataset/chained_transfer/final_model.keras"),
                os.path.join(self.model_dir, "multi_dataset/UNSW-NB15_to_CIC-DDoS19/final_model.keras"),
                os.path.join(self.model_dir, "final_model.keras"),
                os.path.join(self.model_dir, "best_model.keras")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("No model file found. Please specify a valid model path.")
        
        # Locate the scaler if not specified
        if scaler_path is None:
            model_dir = os.path.dirname(model_path)
            possible_paths = [
                os.path.join(model_dir, "scaler.joblib"),
                os.path.join(model_dir, "scaler.pkl"),
                os.path.join(self.model_dir, "scaler.joblib"),
                os.path.join(self.model_dir, "scaler.pkl")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    scaler_path = path
                    break
            
            if scaler_path is None:
                logging.warning("No scaler file found. Predictions may be inaccurate.")
        
        # Load feature mapping if available
        self.feature_map = None
        if feature_map_path is None:
            default_map_path = os.path.join(self.model_dir, "feature_map.json")
            if os.path.exists(default_map_path):
                feature_map_path = default_map_path
        
        if feature_map_path and os.path.exists(feature_map_path):
            try:
                with open(feature_map_path, 'r') as f:
                    self.feature_map = json.load(f)
                logging.info(f"Loaded feature map from {feature_map_path}")
            except Exception as e:
                logging.warning(f"Failed to load feature map: {str(e)}")
        
        # Set up for ensemble if requested
        self.ensemble_models = []
        if use_ensemble:
            # If no ensemble paths provided, create a default set
            if ensemble_paths is None:
                model_dir = os.path.dirname(model_path)
                # Use any other models in the same directory, plus the original
                ensemble_paths = [model_path]
                
                # Add models from alternate datasets if they exist
                alt_model_paths = [
                    os.path.join(self.model_dir, "multi_dataset/UNSW-NB15/best_model.keras"),
                    os.path.join(self.model_dir, "multi_dataset/CIC-DDoS19/best_model.keras"),
                    os.path.join(self.model_dir, "multi_dataset/CSE-CIC-IDS2018/best_model.keras")
                ]
                
                for path in alt_model_paths:
                    if os.path.exists(path) and path not in ensemble_paths:
                        ensemble_paths.append(path)
            
            # Load ensemble models
            for path in ensemble_paths:
                try:
                    if os.path.exists(path):
                        m = tf.keras.models.load_model(path)
                        self.ensemble_models.append(m)
                        logging.info(f"Added model to ensemble: {path}")
                except Exception as e:
                    logging.warning(f"Failed to load ensemble model {path}: {str(e)}")
        
        # Load the primary model
        logging.info(f"Loading model from {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            logging.info(f"Model loaded successfully")
            # Set flag to indicate model is loaded
            self.loaded = True
        except Exception as e:
            self.loaded = False
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Load the scaler if available
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            logging.info(f"Loading scaler from {scaler_path}")
            try:
                self.scaler = joblib.load(scaler_path)
                logging.info(f"Scaler loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load scaler: {str(e)}")
        
        # Store model metadata
        self.input_shape = self.model.input_shape[1:]
        self.n_features = self.input_shape[0] if len(self.input_shape) > 0 else 0
        logging.info(f"Model expects {self.n_features} features")
        
        # Feature importance cache for explanations
        self.feature_importance = self.feature_engineer.feature_importance
    
    def _preprocess(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess the input data for prediction.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Preprocessed data as numpy array
        """
        # Apply feature engineering to enhance detection
        enhanced_data = self.feature_engineer.enhance_features(data)
        
        # Convert input to pandas DataFrame
        if isinstance(enhanced_data, dict):
            df = pd.DataFrame([enhanced_data])
        elif isinstance(enhanced_data, pd.DataFrame):
            df = enhanced_data
        else:
            raise ValueError(f"Unsupported data type: {type(enhanced_data)}")
        
        # Apply feature mapping if available
        if self.feature_map:
            mapped_df = pd.DataFrame()
            for col in df.columns:
                if col in self.feature_map:
                    mapped_df[self.feature_map[col]] = df[col]
                else:
                    mapped_df[col] = df[col]
            df = mapped_df
        
        # Keep track of the base feature columns (not derived)
        base_columns = [f"feature_{i}" for i in range(8)]
        derived_columns = [col for col in df.columns if col.startswith("derived_")]
        all_columns = base_columns + derived_columns
        
        # Convert to numeric and handle NaN values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        
        # Extract only the core model features for scaling
        core_features = df[base_columns].values
        
        # Apply scaling if scaler is available
        if self.scaler:
            try:
                scaled_features = self.scaler.transform(core_features)
            except Exception as e:
                logging.warning(f"Error applying scaler: {str(e)}. Continuing without scaling.")
                scaled_features = core_features
        else:
            scaled_features = core_features
        
        # If we have derived features, we need to handle them separately
        if derived_columns:
            # Store the derived features separately as they are already engineered
            derived_values = df[derived_columns].values
            # For models that only accept base features, we'll return just the scaled features
            # The derived features will be used in ensemble or post-processing
            return scaled_features, derived_values
        
        return scaled_features
    
    def _ensemble_predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make a prediction using all models in the ensemble.
        
        Args:
            features: Preprocessed features
            
        Returns:
            Ensemble prediction probability and confidence
        """
        # If no ensemble models, just use the primary model
        if not self.ensemble_models:
            return self._single_model_predict(self.model, features)
        
        # Get predictions from all models
        predictions = []
        confidences = []
        
        # Get prediction from primary model
        primary_pred, primary_conf = self._single_model_predict(self.model, features)
        predictions.append(primary_pred)
        confidences.append(primary_conf)
        
        # Get predictions from ensemble models
        for model in self.ensemble_models:
            pred, conf = self._single_model_predict(model, features)
            predictions.append(pred)
            confidences.append(conf)
        
        # Calculate weighted average prediction
        weights = np.array(confidences)  # Use confidences as weights
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Calculate weighted prediction
        weighted_pred = np.sum(np.array(predictions) * weights)
        
        # Calculate ensemble confidence
        # Consider both prediction agreement and individual confidences
        pred_std = np.std(predictions)
        avg_conf = np.mean(confidences)
        
        # Higher confidence when predictions agree and individual confidences are high
        ensemble_confidence = avg_conf * (1 - pred_std)
        
        return weighted_pred, ensemble_confidence
    
    def _single_model_predict(self, model: tf.keras.Model, features: np.ndarray) -> Tuple[float, float]:
        """
        Make a prediction using a single model.
        
        Args:
            model: TensorFlow model
            features: Preprocessed features
            
        Returns:
            Prediction probability and confidence
        """
        prediction = model.predict(features, verbose=0)
        
        # Extract probability and confidence
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            # Multi-class, get highest probability
            pred_class = np.argmax(prediction[0])
            confidence = float(prediction[0][pred_class])
            probability = float(prediction[0][1] if pred_class == 1 else 1 - prediction[0][0])
        else:
            # Binary classification
            probability = float(prediction[0][0])
            
            # Calculate confidence based on multiple factors
            # 1. Distance from threshold
            distance = abs(probability - self.base_threshold)
            threshold_confidence = distance / (1 - self.base_threshold)
            
            # 2. Feature anomaly scores
            feature_scores = []
            for i, value in enumerate(features[0]):
                feature_name = f"feature_{i}"
                importance = self.feature_engineer.feature_importance.get(feature_name, 0.5)
                # Calculate how far the value is from the mean (assuming normalized features)
                anomaly_score = abs(value - 0.5) * 2  # Assuming features are normalized to [0,1]
                feature_scores.append(anomaly_score * importance)
            
            feature_confidence = np.mean(feature_scores)
            
            # Combine confidences
            confidence = (threshold_confidence + feature_confidence) / 2
        
        return probability, confidence
    
    def _calculate_dynamic_threshold(self, features: np.ndarray) -> float:
        """
        Calculate a dynamic threshold based on feature patterns.
        
        Args:
            features: Preprocessed features
            
        Returns:
            Dynamic threshold value
        """
        # Base threshold
        base_threshold = 0.15
        
        # Calculate feature statistics
        feature_stats = {
            'mean': np.mean(features),
            'std': np.std(features),
            'max': np.max(features),
            'min': np.min(features)
        }
        
        # Adjust threshold based on feature patterns
        if feature_stats['std'] > 0.5:  # High variance
            base_threshold *= 1.2
        elif feature_stats['std'] < 0.1:  # Low variance
            base_threshold *= 0.8
            
        if feature_stats['max'] > 0.8:  # Extreme values
            base_threshold *= 1.1
            
        # Ensure threshold stays within reasonable bounds
        return max(0.1, min(0.3, base_threshold))
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single sample.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Dictionary with prediction result
        """
        # Get enhanced features first
        enhanced_data = self.feature_engineer.enhance_features(data)
        
        # Get anomaly scores
        anomaly_scores = [v for k, v in enhanced_data.items() if k.endswith("_anomaly")]
        overall_anomaly = np.mean(anomaly_scores) if anomaly_scores else 0.5
        
        # Preprocess data
        features = self._preprocess(enhanced_data)
        
        # If we have derived features, unpack them
        derived_values = None
        if isinstance(features, tuple):
            features, derived_values = features
        
        # Step 1: Get base prediction from primary model
        base_prob, base_conf = self._single_model_predict(self.model, features)
        
        # Step 2: Get ensemble predictions if available
        ensemble_probs = []
        ensemble_confs = []
        if self.use_ensemble and self.ensemble_models:
            for model in self.ensemble_models:
                prob, conf = self._single_model_predict(model, features)
                ensemble_probs.append(prob)
                ensemble_confs.append(conf)
        
        # Step 3: Calculate probability using weighted voting with multiple methods
        if ensemble_probs:
            # Add base model probability
            all_probs = [base_prob] + ensemble_probs
            all_confs = [base_conf] + ensemble_confs
            
            # Method 1: Weighted by confidence
            weights = np.array(all_confs)
            weights = weights / np.sum(weights)  # Normalize
            weighted_prob = np.sum(np.array(all_probs) * weights)
            
            # Method 2: Median probability to avoid outliers
            median_prob = np.median(all_probs)
            
            # Method 3: Mode-like approach using binning
            binned_votes = np.zeros(10)
            for p in all_probs:
                bin_idx = min(9, int(p * 10))
                binned_votes[bin_idx] += 1
            mode_bin = np.argmax(binned_votes)
            mode_prob = (mode_bin + 0.5) / 10  # Center of bin
            
            # Blend methods based on agreement
            variance = np.var(all_probs)
            agreement = 1.0 / (1.0 + 10.0 * variance)
            
            if agreement > 0.8:  # High agreement, trust weighted average
                raw_probability = weighted_prob
            elif agreement < 0.3:  # Low agreement, trust median more
                raw_probability = 0.7 * median_prob + 0.2 * weighted_prob + 0.1 * mode_prob
            else:  # Moderate agreement, balanced blend
                raw_probability = 0.4 * weighted_prob + 0.4 * median_prob + 0.2 * mode_prob
            
            # Calculate agreement-weighted confidence
            confidence = np.mean(all_confs) * (0.5 + 0.5 * agreement)
        else:
            # Only using base model
            raw_probability = base_prob
            confidence = base_conf
            agreement = 1.0
        
        # Step 4: Apply advanced probability calibration
        calibrated_probability = self.probability_calibrator.calibrate(
            raw_probability, 
            anomaly_score=overall_anomaly,
            method='ensemble'  # Use ensemble of calibration methods
        )
        
        # Step 5: Apply multi-criteria decision making
        is_attack, effective_threshold, decision_factors = self.threshold_decision_maker.make_decision(
            probability=calibrated_probability,
            anomaly_score=overall_anomaly,
            confidence=confidence,
            features=features[0] if isinstance(features, np.ndarray) and len(features) > 0 else None,
            derived_features=derived_values[0] if derived_values is not None else None
        )
        
        # Step 6: Calibrate confidence if requested
        if self.calibrate_confidence:
            calibrated_confidence = self.confidence_calibrator.calibrate(confidence)
        else:
            calibrated_confidence = confidence
        
        # Step 7: Calculate risk score combining multiple factors
        # Blend probability, anomaly score, and attack pattern evidence
        pattern_score = decision_factors.get("attack_pattern_score", 0.5)
        risk_factors = [
            calibrated_probability * 0.5,  # Model prediction (strongest weight)
            overall_anomaly * 0.3,         # Anomaly detection
            pattern_score * 0.2            # Attack pattern detection
        ]
        risk_score = min(100, int(sum(risk_factors) * 100))
        
        # Create result dictionary with rich information
        result = {
            "prediction": "attack" if is_attack else "benign",
            "confidence": float(calibrated_confidence),
            "probability": float(calibrated_probability),
            "raw_probability": float(raw_probability),
            "threshold": float(effective_threshold),
            "is_attack": is_attack,
            "risk_score": int(risk_score),
            "anomaly_score": float(overall_anomaly),
            "agreement": float(agreement),
            "decision_factors": decision_factors
        }
        
        # Add derived feature information if available
        if derived_values is not None:
            result["derived_features_used"] = True
        
        return result
    
    def predict_batch(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of samples.
        
        Args:
            data_batch: List of dictionaries containing feature values
            
        Returns:
            List of dictionaries with prediction results
        """
        # Convert batch to DataFrame
        df = pd.DataFrame(data_batch)
        
        # Apply feature engineering to the entire DataFrame at once
        enhanced_df = self.feature_engineer.enhance_features(df)
        
        # Preprocess batch
        features = self._preprocess(enhanced_df)
        
        # If we have derived features, unpack them
        derived_values = None
        if isinstance(features, tuple):
            features, derived_values = features
        
        # Make predictions (using ensemble if enabled)
        results = []
        
        if self.use_ensemble:
            # Process each sample through the ensemble
            for i in range(len(features)):
                probability, confidence = self._ensemble_predict(features[i:i+1])
                
                # Determine prediction label
                pred_label = "attack" if probability >= self.base_threshold else "benign"
                
                # Calibrate confidence if requested
                if self.calibrate_confidence:
                    confidence = self.confidence_calibrator.calibrate(confidence)
                
                # Calculate risk score (scaled 0-100)
                risk_score = min(100, int(probability * 100))
                
                # Create result dictionary
                result = {
                    "prediction": pred_label,
                    "confidence": confidence,
                    "probability": probability,
                    "threshold": self.base_threshold,
                    "is_attack": pred_label == "attack",
                    "risk_score": risk_score
                }
                
                # Add derived feature information if available
                if derived_values is not None:
                    result["derived_features_used"] = True
                
                results.append(result)
        else:
            # Use primary model for all samples at once
            predictions = self.model.predict(features, verbose=0)
            
            # Process results
            for i, pred in enumerate(predictions):
                if len(pred.shape) > 0 and pred.shape[0] > 1:
                    # Multi-class
                    pred_class = np.argmax(pred)
                    confidence = float(pred[pred_class])
                    probability = float(pred[1] if pred_class == 1 else 1 - pred[0])
                else:
                    # Binary
                    probability = float(pred[0])
                    confidence = probability if probability >= self.base_threshold else (1 - probability)
                
                # Determine prediction label
                pred_label = "attack" if probability >= self.base_threshold else "benign"
                
                # Calibrate confidence if requested
                if self.calibrate_confidence:
                    confidence = self.confidence_calibrator.calibrate(confidence)
                
                # Calculate risk score (scaled 0-100)
                risk_score = min(100, int(probability * 100))
                
                # Create result dictionary
                result = {
                    "prediction": pred_label,
                    "confidence": confidence,
                    "probability": probability,
                    "threshold": self.base_threshold,
                    "is_attack": pred_label == "attack",
                    "risk_score": risk_score
                }
                
                # Add derived feature information if available
                if derived_values is not None:
                    result["derived_features_used"] = True
                
                results.append(result)
        
        return results
    
    def explain(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide an explanation for a prediction.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Dictionary with prediction result and explanation
        """
        # First get the prediction using our enhanced prediction logic
        result = self.predict(data)
        
        # Apply feature engineering to enhance detection
        enhanced_data = self.feature_engineer.enhance_features(data)
        
        # Create a DataFrame for feature analysis
        df = pd.DataFrame([enhanced_data])
        
        # Apply preprocessing to get the actual values used for prediction
        features = self._preprocess(df)
        
        # If we have derived features, unpack them
        derived_values = None
        if isinstance(features, tuple):
            features, derived_values = features
        
        processed_features = features[0]
        
        # Map the features back to the original feature names
        feature_values = {}
        for i, value in enumerate(processed_features):
            feature_name = f"feature_{i}"
            feature_values[feature_name] = float(value)
        
        # Add derived features if available
        if derived_values is not None:
            derived_dict = {}
            derived_feature_names = [col for col in df.columns if col.startswith("derived_")]
            for i, name in enumerate(derived_feature_names):
                if i < derived_values.shape[1]:
                    derived_dict[name] = float(derived_values[0, i])
            feature_values.update(derived_dict)
        
        # Add anomaly features
        anomaly_features = {k: v for k, v in enhanced_data.items() if k.endswith("_anomaly")}
        feature_values.update(anomaly_features)
        
        # Add feature values to the result
        result["feature_values"] = feature_values
        
        # Identify abnormal values based on feature importance
        abnormal_features = []
        feature_importance = self.feature_engineer.feature_importance
        
        # Identify the most important contributing features for the prediction
        # We'll select based on a combination of importance and actual value
        contributing_features = []
        
        # For base features
        for name, value in data.items():
            if name.startswith("feature_"):
                importance = feature_importance.get(name, 0.3)
                contribution = abs(value) * importance
                contributing_features.append((name, value, importance, contribution))
        
        # For derived features (give them slightly higher weight)
        for name, value in enhanced_data.items():
            if name.startswith("derived_") and not name.endswith("_anomaly"):
                # Estimate importance
                importance = 0.7
                contribution = abs(value) * importance
                contributing_features.append((name, value, importance, contribution))
                
        # For anomaly features (give them the highest weight)
        for name, value in anomaly_features.items():
            importance = 0.8
            contribution = value * importance  # These are already scaled 0-1
            contributing_features.append((name, value, importance, contribution))
                
        # Sort by contribution and take top contributors
        contributing_features.sort(key=lambda x: x[3], reverse=True)
        
        # Take top contributors
        for name, value, importance, contribution in contributing_features[:5]:
            abnormal_features.append({
                "name": name,
                "value": value,
                "importance": importance,
                "contribution": contribution
            })
        
        # Sort abnormal features by contribution
        abnormal_features.sort(key=lambda x: x.get("contribution", 0), reverse=True)
        
        # Add abnormal features to the result
        result["abnormal_features"] = abnormal_features
        
        # Add threat intelligence (summary of the detected threat)
        if result["is_attack"]:
            # Determine likely attack type based on feature patterns
            attack_type = self._determine_attack_type(data, enhanced_data)
            result["threat_intelligence"] = {
                "attack_type": attack_type,
                "severity": "high" if result["risk_score"] > 70 else "medium" if result["risk_score"] > 40 else "low",
                "confidence": result["confidence"],
                "mitigations": self._get_mitigations(attack_type)
            }
        
        return result
    
    def _determine_attack_type(self, data: Dict[str, Any], enhanced_data: Dict[str, Any]) -> str:
        """
        Determine the likely attack type based on feature patterns.
        
        Args:
            data: Dictionary containing original feature values
            enhanced_data: Dictionary containing enhanced feature values
            
        Returns:
            String describing the likely attack type
        """
        # Extract key features
        flow_duration = data.get("feature_0", 0)
        byte_count = data.get("feature_1", 0)
        packet_count = data.get("feature_2", 0)
        byte_rate = data.get("feature_3", 0)
        ttl = data.get("feature_4", 0)
        window_size = data.get("feature_5", 0)
        packet_size = data.get("feature_6", 0)
        inter_arrival = data.get("feature_7", 0)
        
        # Extract derived features if available
        packets_per_second = enhanced_data.get("derived_packets_per_second", 0)
        bytes_per_packet = enhanced_data.get("derived_bytes_per_packet", 0)
        burst_ratio = enhanced_data.get("derived_burst_ratio", 0)
        
        # Check for DDoS patterns
        if packets_per_second > 1000 or burst_ratio > 5000:
            if bytes_per_packet < 100:
                return "SYN Flood DDoS"
            elif bytes_per_packet < 200:
                return "UDP Flood DDoS"
            else:
                return "HTTP Flood DDoS"
        
        # Check for port scanning
        if packet_count > 100 and packet_size < 100 and flow_duration < 10:
            return "Port Scanning"
        
        # Check for brute force
        if packet_count > 50 and 200 < byte_rate < 1000 and 40 < packet_size < 200:
            return "Brute Force Attack"
        
        # Check for data exfiltration
        if byte_count > 10000 and bytes_per_packet > 1000:
            return "Data Exfiltration"
        
        # Default to generic classification based on features
        if byte_rate > 5000:
            return "Volumetric Attack"
        elif packet_count > 500:
            return "Protocol Attack"
        else:
            return "Application Layer Attack"
    
    def _get_mitigations(self, attack_type: str) -> List[str]:
        """
        Get suggested mitigations for a detected attack type.
        
        Args:
            attack_type: String describing the attack type
            
        Returns:
            List of mitigation strategies
        """
        # Define mitigations for common attack types
        mitigations = {
            "SYN Flood DDoS": [
                "Enable SYN cookies",
                "Implement rate limiting at the network edge",
                "Deploy DDoS protection service"
            ],
            "UDP Flood DDoS": [
                "Configure firewall to rate limit UDP traffic",
                "Deploy traffic scrubbing service",
                "Implement source IP verification"
            ],
            "HTTP Flood DDoS": [
                "Deploy web application firewall (WAF)",
                "Implement CAPTCHA for suspicious clients",
                "Enable caching for static content"
            ],
            "Port Scanning": [
                "Configure firewall to block IPs after multiple connection attempts",
                "Implement port knocking for sensitive services",
                "Hide services behind VPN or bastion host"
            ],
            "Brute Force Attack": [
                "Implement account lockout policies",
                "Enable multi-factor authentication",
                "Use strong password policy"
            ],
            "Data Exfiltration": [
                "Monitor and limit outbound data transfers",
                "Implement data loss prevention (DLP) solution",
                "Encrypt sensitive data at rest"
            ],
            "Volumetric Attack": [
                "Increase bandwidth capacity",
                "Deploy traffic scrubbing service",
                "Implement anycast network addressing"
            ],
            "Protocol Attack": [
                "Configure protocol-specific filtering rules",
                "Update network infrastructure firmware",
                "Deploy intrusion prevention system (IPS)"
            ],
            "Application Layer Attack": [
                "Deploy web application firewall (WAF)",
                "Implement rate limiting for API endpoints",
                "Perform regular security scans and updates"
            ]
        }
        
        # Return mitigations for the specific attack type, or generic mitigations
        return mitigations.get(attack_type, [
            "Monitor network traffic for anomalies",
            "Update security systems and patches",
            "Implement layered security measures"
        ])

# For direct testing
if __name__ == "__main__":
    # Create predictor with optimal threshold and all improvements
    predictor = NeuraShieldPredictor(
        threshold=0.1,
        calibrate_confidence=True,
        use_ensemble=True
    )
    
    # Generate a more realistic attack sample
    attack_sample = {
        "feature_0": 0.002,           # Very short flow duration (suspicious)
        "feature_1": 75,              # Small bytes transferred
        "feature_2": 800,             # Many packets (suspicious)
        "feature_3": 9500,            # Very high byte rate (suspicious)
        "feature_4": 5,               # Abnormal TTL (suspicious)
        "feature_5": 50,              # Small window size (suspicious)
        "feature_6": 25,              # Tiny packet size (suspicious)
        "feature_7": 0.001            # Very small inter-arrival time (suspicious)
    }
    
    # Generate a realistic benign sample
    benign_sample = {
        "feature_0": 0.5,             # Normal flow duration
        "feature_1": 8000,            # Normal bytes transferred
        "feature_2": 20,              # Normal packet count
        "feature_3": 500,             # Normal byte rate
        "feature_4": 64,              # Standard TTL
        "feature_5": 16384,           # Normal window size
        "feature_6": 1200,            # Normal packet size
        "feature_7": 0.02             # Normal inter-arrival time
    }
    
    # Test attack sample
    attack_result = predictor.predict(attack_sample)
    print(f"Attack sample prediction: {attack_result['prediction']} (risk score: {attack_result['risk_score']})")
    
    # Test benign sample
    benign_result = predictor.predict(benign_sample)
    print(f"Benign sample prediction: {benign_result['prediction']} (risk score: {benign_result['risk_score']})")
    
    # Get explanation for attack sample
    explanation = predictor.explain(attack_sample)
    abnormal = explanation.get("abnormal_features", [])
    
    if abnormal:
        print("\nAbnormal features in attack sample:")
        for feature in abnormal:
            print(f"- {feature['name']}: {feature['value']} (importance: {feature['importance']:.2f})")
    
    # Get threat intelligence
    if "threat_intelligence" in explanation:
        ti = explanation["threat_intelligence"]
        print(f"\nThreat intelligence:")
        print(f"- Attack type: {ti['attack_type']}")
        print(f"- Severity: {ti['severity']}")
        print(f"- Confidence: {ti['confidence']:.2f}")
        print(f"\nRecommended mitigations:")
        for m in ti['mitigations']:
            print(f"- {m}") 