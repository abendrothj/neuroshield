#!/usr/bin/env python3

"""
NeuraShield Ensemble Models
This module provides ensemble methods for combining multiple threat detection models
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import logging
from typing import List, Dict, Tuple, Union, Optional
import pickle

from advanced_models import build_residual_nn, build_conv_nn, build_sequential_nn

class EnsembleModel:
    """
    Ensemble model that combines multiple neural networks
    """
    
    def __init__(self, input_shape: int, num_classes: int, models_config: Optional[List[Dict]] = None):
        """
        Initialize the ensemble model
        
        Args:
            input_shape: Number of input features
            num_classes: Number of output classes
            models_config: List of model configurations
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
        self.meta_model = None
        
        # Default model configuration if none provided
        if models_config is None:
            self.models_config = [
                {"type": "residual", "params": {"units": 128, "num_blocks": 3}},
                {"type": "residual", "params": {"units": 256, "num_blocks": 2}},
                {"type": "conv", "params": {"sequence_length": 15}}
            ]
        else:
            self.models_config = models_config
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual models based on configuration"""
        for i, config in enumerate(self.models_config):
            model_type = config["type"]
            params = config.get("params", {})
            
            if model_type == "residual":
                model = build_residual_nn(
                    self.input_shape, 
                    self.num_classes,
                    **params
                )
            elif model_type == "conv":
                # Remove the CNN model from the ensemble to avoid shape issues
                # Instead, use a different residual model with varied parameters
                model = build_residual_nn(
                    self.input_shape, 
                    self.num_classes,
                    units=256 if i % 2 == 0 else 128,
                    num_blocks=2 if i % 2 == 0 else 3
                )
            elif model_type == "sequential":
                # For sequential models, we'll also use a residual network
                # to avoid shape mismatch issues
                model = build_residual_nn(
                    self.input_shape, 
                    self.num_classes,
                    units=192,
                    num_blocks=2
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model._name = f"{model_type}_model_{i}"
            self.models.append(model)
            
            logging.info(f"Initialized {model._name}")
    
    def _build_meta_model(self, num_models=None):
        """
        Build meta-model for combining base model predictions
        
        Args:
            num_models: Number of models to combine
        """
        if num_models is None:
            num_models = len(self.models)
        
        # Each base model produces probabilities for each class
        meta_inputs = Input(shape=(num_models * self.num_classes,))
        
        # Meta-model is a simple neural network
        x = Dense(64, activation='relu')(meta_inputs)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        meta_outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.meta_model = Model(meta_inputs, meta_outputs)
        self.meta_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']  # Removed AUC metric to fix dimension mismatch issues
        )
        
        logging.info("Meta-model initialized")
    
    def train_base_models(self, X, y, validation_data=None, epochs=50, batch_size=64, callbacks=None):
        """
        Train all base models
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of callbacks
            
        Returns:
            List of training histories
        """
        histories = []
        
        for i, model in enumerate(self.models):
            logging.info(f"Training model {i+1}/{len(self.models)}: {model._name}")
            
            history = model.fit(
                X, y,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            histories.append(history)
            
            # Save individual model predictions for meta-model training
            logging.info(f"Base model {i+1} trained")
        
        return histories
    
    def _get_meta_features(self, X):
        """
        Get meta-features by combining predictions from base models
        
        Args:
            X: Input features
            
        Returns:
            Meta-features (combined predictions)
        """
        # Get predictions from each base model
        all_preds = []
        for model in self.models:
            preds = model.predict(X)
            all_preds.append(preds)
        
        # Combine predictions into a single feature array
        meta_features = np.concatenate(all_preds, axis=1)
        return meta_features
    
    def train_meta_model(self, X, y, validation_data=None, epochs=20, batch_size=64):
        """
        Train the meta-model using base model predictions
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Meta-model training history
        """
        if self.meta_model is None:
            self._build_meta_model()
        
        # Get meta-features
        meta_features = self._get_meta_features(X)
        
        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            meta_val = self._get_meta_features(X_val)
            validation_data = (meta_val, y_val)
        
        # Train meta-model
        history = self.meta_model.fit(
            meta_features, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        logging.info("Meta-model trained")
        return history
    
    def predict(self, X):
        """
        Make predictions using the ensemble model
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Get meta-features for meta-model prediction
        meta_features = self._get_meta_features(X)
        
        # Get predictions from meta-model if it exists
        # Otherwise use average predictions from base models
        if hasattr(self, 'meta_model') and self.meta_model is not None:
            return self.meta_model.predict(meta_features)
        else:
            # Average base model predictions
            all_preds = []
            for model in self.models:
                preds = model.predict(X)
                all_preds.append(preds)
            
            # Average predictions
            return np.mean(all_preds, axis=0)
    
    def save(self, save_dir):
        """
        Save the ensemble model
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save base models
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f"base_model_{i}.keras")
            model.save(model_path)
        
        # Save meta-model if trained
        if self.meta_model is not None:
            meta_model_path = os.path.join(save_dir, "meta_model.keras")
            self.meta_model.save(meta_model_path)
        
        # Save configuration
        config = {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "models_config": self.models_config,
            "num_models": len(self.models)
        }
        
        with open(os.path.join(save_dir, "ensemble_config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
        logging.info(f"Ensemble model saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir):
        """
        Load ensemble model from directory
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            Loaded ensemble model
        """
        # Load configuration
        with open(os.path.join(load_dir, "ensemble_config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Create instance
        ensemble = cls(
            input_shape=config["input_shape"],
            num_classes=config["num_classes"],
            models_config=config["models_config"]
        )
        
        # Clear initialized models
        ensemble.models = []
        
        # Load base models
        for i in range(config["num_models"]):
            model_path = os.path.join(load_dir, f"base_model_{i}.keras")
            if os.path.exists(model_path):
                model = load_model(model_path, compile=False)
                ensemble.models.append(model)
        
        # Load meta-model if exists
        meta_model_path = os.path.join(load_dir, "meta_model.keras")
        if os.path.exists(meta_model_path):
            ensemble.meta_model = load_model(meta_model_path, compile=False)
        
        logging.info(f"Ensemble model loaded from {load_dir}")
        return ensemble


class HybridEnsemble:
    """
    Hybrid ensemble that combines neural networks with traditional ML models
    """
    
    def __init__(self, input_shape, num_classes):
        """
        Initialize the hybrid ensemble
        
        Args:
            input_shape: Number of input features
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.nn_model = None
        self.rf_model = None
        self.gb_model = None
        self.voting_weights = [0.4, 0.3, 0.3]  # Default weights for [NN, RF, GB]
    
    def train(self, X, y, validation_data=None, epochs=50, batch_size=64):
        """
        Train all models in the hybrid ensemble
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Validation data for neural network
            epochs: Number of epochs for neural network
            batch_size: Batch size for neural network
        """
        # Train neural network
        self.nn_model = build_residual_nn(self.input_shape, self.num_classes)
        
        if validation_data is not None:
            X_val, y_val = validation_data
        else:
            X_val, y_val = None, None
        
        self.nn_model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)
        
        # Train Gradient Boosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.gb_model.fit(X, y)
        
        # Optimize voting weights if validation data provided
        if validation_data is not None:
            self._optimize_weights(X_val, y_val)
        
        logging.info("Hybrid ensemble trained successfully")
    
    def _optimize_weights(self, X_val, y_val):
        """
        Optimize voting weights using validation data
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        # Get predictions from each model
        nn_preds = self.nn_model.predict(X_val)
        rf_preds = self.rf_model.predict_proba(X_val)
        gb_preds = self.gb_model.predict_proba(X_val)
        
        # Simple grid search for weights
        best_accuracy = 0
        best_weights = self.voting_weights
        
        # Try different weight combinations
        for w1 in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for w2 in [0.2, 0.3, 0.4]:
                w3 = 1.0 - w1 - w2
                if w3 <= 0:
                    continue
                
                weights = [w1, w2, w3]
                weighted_preds = (
                    w1 * nn_preds + 
                    w2 * rf_preds + 
                    w3 * gb_preds
                )
                
                pred_classes = np.argmax(weighted_preds, axis=1)
                accuracy = np.mean(pred_classes == y_val)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights
        
        self.voting_weights = best_weights
        logging.info(f"Optimized weights: {self.voting_weights}")
    
    def predict(self, X):
        """
        Make predictions using weighted voting
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        # Get predictions from each model
        nn_preds = self.nn_model.predict(X)
        rf_preds = self.rf_model.predict_proba(X)
        gb_preds = self.gb_model.predict_proba(X)
        
        # Weighted average
        w1, w2, w3 = self.voting_weights
        weighted_preds = w1 * nn_preds + w2 * rf_preds + w3 * gb_preds
        
        return weighted_preds
    
    def save(self, save_dir):
        """
        Save the hybrid ensemble
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save neural network
        if self.nn_model is not None:
            self.nn_model.save(os.path.join(save_dir, "nn_model.keras"))
        
        # Save Random Forest
        if self.rf_model is not None:
            joblib.dump(self.rf_model, os.path.join(save_dir, "rf_model.joblib"))
        
        # Save Gradient Boosting
        if self.gb_model is not None:
            joblib.dump(self.gb_model, os.path.join(save_dir, "gb_model.joblib"))
        
        # Save configuration
        config = {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "voting_weights": self.voting_weights
        }
        
        with open(os.path.join(save_dir, "hybrid_config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
        logging.info(f"Hybrid ensemble saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir):
        """
        Load hybrid ensemble from directory
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            Loaded hybrid ensemble
        """
        # Load configuration
        with open(os.path.join(load_dir, "hybrid_config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Create instance
        ensemble = cls(
            input_shape=config["input_shape"],
            num_classes=config["num_classes"]
        )
        
        # Set weights
        ensemble.voting_weights = config["voting_weights"]
        
        # Load neural network
        nn_path = os.path.join(load_dir, "nn_model.keras")
        if os.path.exists(nn_path):
            ensemble.nn_model = load_model(nn_path)
        
        # Load Random Forest
        rf_path = os.path.join(load_dir, "rf_model.joblib")
        if os.path.exists(rf_path):
            ensemble.rf_model = joblib.load(rf_path)
        
        # Load Gradient Boosting
        gb_path = os.path.join(load_dir, "gb_model.joblib")
        if os.path.exists(gb_path):
            ensemble.gb_model = joblib.load(gb_path)
        
        logging.info(f"Hybrid ensemble loaded from {load_dir}")
        return ensemble


class SpecializedEnsemble:
    """
    Ensemble of specialized models for different types of threats
    """
    
    def __init__(self, input_shape, threat_types=None):
        """
        Initialize specialized ensemble
        
        Args:
            input_shape: Number of input features
            threat_types: List of threat types
        """
        self.input_shape = input_shape
        
        # Default threat types if none provided
        if threat_types is None:
            self.threat_types = [
                "Normal", "DoS", "Probe", "R2L", "U2R"
            ]
        else:
            self.threat_types = threat_types
        
        self.num_classes = len(self.threat_types)
        self.specialized_models = {}
        self.meta_model = None
    
    def train_specialized_models(self, X, y, validation_data=None, epochs=50, batch_size=64):
        """
        Train specialized models for each threat type
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Dictionary of training histories
        """
        histories = {}
        
        # Create one-vs-rest dataset for each threat type
        for idx, threat_type in enumerate(self.threat_types):
            if threat_type == "Normal":
                continue  # Skip normal traffic, it's handled by all models
                
            logging.info(f"Training specialized model for {threat_type}")
            
            # Binary labels for this threat (1 for this threat, 0 for others)
            binary_y = (y == idx).astype(int)
            
            # Create binary validation data if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                binary_y_val = (y_val == idx).astype(int)
                binary_val_data = (X_val, binary_y_val)
            else:
                binary_val_data = None
            
            # Build specialized model for this threat
            model = build_residual_nn(self.input_shape, 2)  # Binary classification
            
            # Train model
            history = model.fit(
                X, binary_y,
                validation_data=binary_val_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Store model and history
            self.specialized_models[threat_type] = model
            histories[threat_type] = history
            
            logging.info(f"Specialized model for {threat_type} trained")
        
        return histories
    
    def _build_meta_model(self):
        """Build meta-model for combining specialized model predictions"""
        # Input shape is sum of all specialized model outputs (each gives 2 probabilities)
        meta_input_shape = len(self.specialized_models) * 2
        
        # Build meta-model
        meta_inputs = Input(shape=(meta_input_shape,))
        x = Dense(64, activation='relu')(meta_inputs)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        meta_outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.meta_model = Model(meta_inputs, meta_outputs)
        self.meta_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']  # Using only accuracy to avoid dimension issues
        )
        
        logging.info("Meta-model initialized")
    
    def _get_meta_features(self, X):
        """
        Get meta-features by combining predictions from specialized models
        
        Args:
            X: Input features
            
        Returns:
            Meta-features (combined predictions)
        """
        # Get predictions from each specialized model
        all_preds = []
        for threat_type, model in self.specialized_models.items():
            preds = model.predict(X)
            all_preds.append(preds)
        
        # Combine predictions into a single feature array
        meta_features = np.concatenate(all_preds, axis=1)
        return meta_features
    
    def train_meta_model(self, X, y, validation_data=None, epochs=20, batch_size=64):
        """
        Train the meta-model using specialized model predictions
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Meta-model training history
        """
        if self.meta_model is None:
            self._build_meta_model()
        
        # Get meta-features
        meta_features = self._get_meta_features(X)
        
        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            meta_val = self._get_meta_features(X_val)
            validation_data = (meta_val, y_val)
        
        # Train meta-model
        history = self.meta_model.fit(
            meta_features, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        logging.info("Meta-model trained")
        return history
    
    def predict(self, X):
        """
        Make predictions using the specialized ensemble
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.specialized_models:
            raise ValueError("No specialized models trained")
        
        if self.meta_model is None:
            # If meta-model not trained, use max probability across specialized models
            all_probs = np.zeros((X.shape[0], self.num_classes))
            
            # Default probability for normal traffic
            all_probs[:, 0] = 0.5
            
            # Update with specialized model probabilities
            for idx, (threat_type, model) in enumerate(self.specialized_models.items(), 1):
                if idx < self.num_classes:
                    threat_probs = model.predict(X)
                    all_probs[:, idx] = threat_probs[:, 1]  # Probability of being this threat
            
            # Normalize probabilities
            row_sums = all_probs.sum(axis=1, keepdims=True)
            normalized_probs = all_probs / row_sums
            
            return normalized_probs
        else:
            # Get meta-features
            meta_features = self._get_meta_features(X)
            
            # Use meta-model for predictions
            return self.meta_model.predict(meta_features)
    
    def save(self, save_dir):
        """
        Save the specialized ensemble
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save specialized models
        for threat_type, model in self.specialized_models.items():
            model_path = os.path.join(save_dir, f"{threat_type}_model.keras")
            model.save(model_path)
        
        # Save meta-model if trained
        if self.meta_model is not None:
            meta_model_path = os.path.join(save_dir, "meta_model.keras")
            self.meta_model.save(meta_model_path)
        
        # Save configuration
        config = {
            "input_shape": self.input_shape,
            "threat_types": self.threat_types,
            "num_classes": self.num_classes
        }
        
        with open(os.path.join(save_dir, "specialized_config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
        logging.info(f"Specialized ensemble saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir):
        """
        Load specialized ensemble from directory
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            Loaded specialized ensemble
        """
        # Load configuration
        with open(os.path.join(load_dir, "specialized_config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Create instance
        ensemble = cls(
            input_shape=config["input_shape"],
            threat_types=config["threat_types"]
        )
        
        # Load specialized models
        for threat_type in ensemble.threat_types:
            if threat_type == "Normal":
                continue
                
            model_path = os.path.join(load_dir, f"{threat_type}_model.keras")
            if os.path.exists(model_path):
                model = load_model(model_path, compile=False)
                ensemble.specialized_models[threat_type] = model
        
        # Load meta-model if exists
        meta_model_path = os.path.join(load_dir, "meta_model.keras")
        if os.path.exists(meta_model_path):
            ensemble.meta_model = load_model(meta_model_path, compile=False)
        
        logging.info(f"Specialized ensemble loaded from {load_dir}")
        return ensemble 