from threat_detection_model import ThreatDetectionModel
import numpy as np

def main():
    # 1. Create a model instance
    print("Creating model instance...")
    model = ThreatDetectionModel(input_shape=(9,), num_classes=5)
    
    # 2. Generate some sample data (replace this with your actual data)
    print("\nGenerating sample data...")
    # Training data
    X_train = np.random.randn(1000, 9)  # 1000 samples, 9 features
    y_train = np.random.randint(0, 5, 1000)  # 1000 labels (0-4)
    
    # Validation data
    X_val = np.random.randn(200, 9)  # 200 samples, 9 features
    y_val = np.random.randint(0, 5, 200)  # 200 labels (0-4)
    
    # 3. Train the model
    print("\nTraining the model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=32
    )
    
    # 4. Make predictions on new data
    print("\nMaking predictions...")
    # Generate some new data (replace with your actual new data)
    X_new = np.random.randn(5, 9)  # 5 new samples
    predictions = model.predict(X_new)
    
    # Print predictions
    print("\nPredictions for new samples:")
    for i, pred in enumerate(predictions):
        predicted_class = np.argmax(pred)
        confidence = np.max(pred) * 100
        print(f"Sample {i+1}: Predicted class = {predicted_class}, Confidence = {confidence:.2f}%")
    
    # 5. Save the model
    print("\nSaving the model...")
    model.save("ai_models/threat_detection")
    
    # 6. Load the model (demonstration)
    print("\nLoading the saved model...")
    loaded_model = ThreatDetectionModel()
    loaded_model.load("ai_models/threat_detection")
    
    # Verify loaded model works
    print("\nVerifying loaded model with predictions...")
    loaded_predictions = loaded_model.predict(X_new)
    print("Loaded model predictions match original model:", np.allclose(predictions, loaded_predictions))

if __name__ == "__main__":
    main() 