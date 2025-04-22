#!/usr/bin/env python3

"""
NeuraShield Advanced Model Architectures
This module provides advanced neural network architectures for threat detection
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, 
    Concatenate, Add, LSTM, GRU, Bidirectional,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam, RMSprop

def residual_block(x, units, dropout_rate=0.3):
    """Create a residual block for neural networks"""
    # Store the input for the residual connection
    residual = x
    
    # First layer
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # If input shape doesn't match output shape, use a projection
    if residual.shape[-1] != units:
        residual = Dense(units, activation='linear')(residual)
    
    # Add the residual connection
    x = Add()([x, residual])
    x = Dropout(dropout_rate)(x)
    
    return x

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss implementation that works in graph mode
    
    Args:
        gamma: Focusing parameter
        alpha: Weighting factor
    
    Returns:
        Focal loss function
    """
    def sparse_categorical_focal_loss(y_true, y_pred):
        """Focal loss function that works with sparse and one-hot labels without conditionals."""
        # Get number of classes from predictions
        num_classes = tf.shape(y_pred)[-1]
        
        # Always treat y_true as sparse and convert to one-hot
        # Reshape to 1D tensor
        y_true_flat = tf.reshape(y_true, [-1])
        y_true_flat = tf.cast(y_true_flat, tf.int32)
        
        # Ensure valid indices by clipping
        y_true_flat = tf.clip_by_value(y_true_flat, 0, num_classes-1)
        
        # Convert to one-hot
        y_true_one_hot = tf.one_hot(y_true_flat, depth=num_classes)
        
        # Reshape back to match y_pred shape
        y_true_one_hot = tf.reshape(y_true_one_hot, tf.shape(y_pred))
        
        # Standard focal loss calculation
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # Calculate the focal term
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Apply alpha if provided
        if alpha is not None:
            # Create alpha tensor with appropriate broadcasting
            alpha_tensor = tf.ones_like(cross_entropy) * alpha
            focal_weight = focal_weight * alpha_tensor
            
        # Apply weight to cross entropy
        loss = focal_weight * cross_entropy
        
        # Sum across classes and average across batch
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        
        return loss
    
    return sparse_categorical_focal_loss

def build_residual_nn(input_shape, num_classes, units=128, num_blocks=3, dropout_rate=0.3):
    """
    Build a residual neural network model
    
    Args:
        input_shape: Input shape (number of features)
        num_classes: Number of output classes
        units: Base number of units
        num_blocks: Number of residual blocks
        dropout_rate: Dropout rate
        
    Returns:
        Compiled model
    """
    inputs = Input(shape=(input_shape,))
    
    # Initial layer
    x = Dense(units, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Residual blocks
    for i in range(num_blocks):
        block_units = units * (2 ** min(i, 2))  # Increase units up to 4x
        x = residual_block(x, block_units, dropout_rate)
    
    # Output layer
    x = Dense(units // 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_sequential_nn(input_shape, num_classes, sequence_length=10):
    """
    Build a sequential neural network (LSTM/GRU) for time series data
    
    Args:
        input_shape: Input shape (number of features)
        num_classes: Number of output classes
        sequence_length: Length of the time sequence
        
    Returns:
        Compiled model
    """
    # Reshape input for sequence model
    # input_shape becomes (sequence_length, features_per_timestep)
    features_per_timestep = input_shape // sequence_length
    if features_per_timestep < 1:
        features_per_timestep = 1
        sequence_length = input_shape
    
    inputs = Input(shape=(sequence_length, features_per_timestep))
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_conv_nn(input_shape, num_classes, sequence_length=10):
    """
    Build a 1D convolutional neural network for pattern detection
    
    Args:
        input_shape: Input shape (number of features)
        num_classes: Number of output classes
        sequence_length: Length of the time sequence
        
    Returns:
        Compiled model
    """
    # Reshape input for CNN model
    features_per_timestep = input_shape // sequence_length
    if features_per_timestep < 1:
        features_per_timestep = 1
        sequence_length = input_shape
    
    inputs = Input(shape=(sequence_length, features_per_timestep))
    
    # Conv1D layers with different kernel sizes to capture different patterns
    conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv2 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    conv3 = Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)
    
    # Combine different convolution results
    x = Concatenate()([conv1, conv2, conv3])
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Additional Conv1D layer
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Global pooling and dense layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    return model

def build_hybrid_nn(input_shape, num_classes, sequence_length=10):
    """
    Build a hybrid neural network combining different architectures
    
    Args:
        input_shape: Input shape (number of features)
        num_classes: Number of output classes
        sequence_length: Length of the time sequence
        
    Returns:
        Compiled model
    """
    # Split input for different model branches
    # Static features and sequential features
    static_features = input_shape // 2
    seq_features = input_shape - static_features
    
    # Ensure minimum dimensions
    static_features = max(static_features, 5)
    seq_features = max(seq_features, 10)
    
    # Reshape sequential features
    features_per_timestep = seq_features // sequence_length
    if features_per_timestep < 1:
        features_per_timestep = 1
        sequence_length = seq_features
    
    # Static features branch
    static_input = Input(shape=(static_features,))
    s = Dense(128, activation='relu')(static_input)
    s = BatchNormalization()(s)
    s = Dropout(0.3)(s)
    s = residual_block(s, 128, 0.3)
    
    # Sequential features branch
    seq_input = Input(shape=(sequence_length, features_per_timestep))
    
    # Conv path
    c = Conv1D(64, kernel_size=3, activation='relu', padding='same')(seq_input)
    c = MaxPooling1D(pool_size=2)(c)
    c = Dropout(0.3)(c)
    c = Conv1D(128, kernel_size=3, activation='relu', padding='same')(c)
    c = GlobalAveragePooling1D()(c)
    
    # LSTM path
    l = Bidirectional(LSTM(64, return_sequences=False))(seq_input)
    l = Dropout(0.3)(l)
    
    # Combine sequential branches
    seq_combined = Concatenate()([c, l])
    seq_combined = Dense(128, activation='relu')(seq_combined)
    seq_combined = BatchNormalization()(seq_combined)
    seq_combined = Dropout(0.3)(seq_combined)
    
    # Combine all branches
    combined = Concatenate()([s, seq_combined])
    combined = Dense(128, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    outputs = Dense(num_classes, activation='softmax')(combined)
    
    # Create and compile model
    model = Model([static_input, seq_input], outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    return model

def build_autoencoder_detector(input_shape, encoding_dim=32, num_classes=2):
    """
    Build an autoencoder-based anomaly detector
    
    Args:
        input_shape: Input shape (number of features)
        encoding_dim: Dimension of the encoded representation
        num_classes: Number of output classes
        
    Returns:
        Tuple of (encoder_model, classifier_model)
    """
    # Encoder
    inputs = Input(shape=(input_shape,))
    encoded = Dense(128, activation='relu')(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    bottleneck = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
    
    # Decoder (for pretraining)
    decoded = Dense(64, activation='relu')(bottleneck)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    outputs_ae = Dense(input_shape, activation='linear')(decoded)
    
    # Classifier using bottleneck features
    classifier = Dense(64, activation='relu')(bottleneck)
    classifier = BatchNormalization()(classifier)
    classifier = Dropout(0.3)(classifier)
    outputs_clf = Dense(num_classes, activation='softmax')(classifier)
    
    # Create autoencoder model for pretraining
    autoencoder = Model(inputs, outputs_ae)
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Create classifier model that includes the encoding part
    classifier_model = Model(inputs, outputs_clf)
    classifier_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    return autoencoder, classifier_model 