from threat_detection_model import ThreatDetectionModel
import numpy as np
from sklearn.model_selection import train_test_split

def generate_threat_data(n_samples=1000):
    """
    Generate realistic training data for threat detection
    Features represent various network/security metrics
    """
    # Define class characteristics
    class_characteristics = {
        0: {'mean': 0.1, 'std': 0.1},    # Normal traffic
        1: {'mean': 0.7, 'std': 0.2},    # DDoS attack
        2: {'mean': 0.8, 'std': 0.15},   # Brute force
        3: {'mean': 0.6, 'std': 0.25},   # Port scan
        4: {'mean': 0.9, 'std': 0.1}     # Malware
    }
    
    # Generate data for each class
    X = []
    y = []
    
    for class_idx, characteristics in class_characteristics.items():
        # Generate samples for this class
        n_class_samples = n_samples // len(class_characteristics)
        
        # Generate base features with class-specific characteristics
        base_features = np.random.normal(
            characteristics['mean'],
            characteristics['std'],
            (n_class_samples, 3)
        )
        
        # Add some noise and variations
        noise = np.random.normal(0, 0.1, (n_class_samples, 6))
        
        # Combine features
        class_features = np.concatenate([base_features, noise], axis=1)
        
        # Ensure values are between 0 and 1
        class_features = np.clip(class_features, 0, 1)
        
        X.append(class_features)
        y.extend([class_idx] * n_class_samples)
    
    return np.array(X).reshape(-1, 9), np.array(y)

def main():
    # Generate training data
    print("Generating training data...")
    X, y = generate_threat_data(n_samples=1000)
    
    # Split into training and validation sets
    print("\nSplitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("\nClass distribution:")
    for i in range(5):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        print(f"Class {i}:")
        print(f"  Training: {train_count} samples")
        print(f"  Validation: {val_count} samples")
    
    # Create and train model
    print("\nCreating and training model...")
    model = ThreatDetectionModel(input_shape=(9,), num_classes=5)
    
    # Train the model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_predictions = model.predict(X_val)
    val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == y_val) * 100
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    
    # Save the model
    print("\nSaving the trained model...")
    model.save("ai_models/threat_detection")
    
    # Test with some example samples
    print("\nTesting with example samples...")
    example_samples = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Normal traffic
        [0.8, 0.7, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # DDoS attack
        [0.9, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Brute force
        [0.7, 0.6, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Port scan
        [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]   # Malware
    ])
    
    predictions = model.predict(example_samples)
    print("\nPredictions for example samples:")
    threat_types = ["Normal", "DDoS", "Brute Force", "Port Scan", "Malware"]
    for i, (pred, true_type) in enumerate(zip(predictions, threat_types)):
        predicted_class = np.argmax(pred)
        confidence = np.max(pred) * 100
        print(f"Sample {i+1} ({true_type}):")
        print(f"  Predicted: {threat_types[predicted_class]}")
        print(f"  Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main() 