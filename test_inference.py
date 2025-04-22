#!/usr/bin/env python3

"""
NeuraShield Inference Test
This script tests the inference engine to ensure it's working correctly.
"""

import os
import sys
import time
import logging
import numpy as np
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the inference module
from inference import NeuraShieldPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def generate_advanced_test_samples(num_samples: int = 50) -> List[Dict[str, float]]:
    """
    Generate a diverse set of test network traffic samples with realistic patterns.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of network traffic samples
    """
    samples = []
    
    # 1. Generate normal web browsing traffic (benign)
    for i in range(int(num_samples * 0.2)):
        sample = {
            "feature_0": np.random.uniform(0.5, 5.0),       # Normal flow duration
            "feature_1": np.random.uniform(1000, 50000),    # Normal data size
            "feature_2": np.random.uniform(5, 50),          # Normal packet count
            "feature_3": np.random.uniform(200, 2000),      # Normal byte rate
            "feature_4": np.random.uniform(50, 64),         # Normal TTL
            "feature_5": np.random.uniform(8000, 65535),    # Normal window size
            "feature_6": np.random.uniform(500, 1460),      # Normal packet size
            "feature_7": np.random.uniform(0.01, 0.2)       # Normal inter-arrival time
        }
        samples.append(sample)
    
    # 2. Generate file download traffic (benign)
    for i in range(int(num_samples * 0.1)):
        sample = {
            "feature_0": np.random.uniform(5.0, 30.0),      # Longer flow duration
            "feature_1": np.random.uniform(5000000, 50000000), # Large data size
            "feature_2": np.random.uniform(1000, 10000),    # Many packets
            "feature_3": np.random.uniform(5000, 20000),    # Higher byte rate
            "feature_4": np.random.uniform(50, 64),         # Normal TTL
            "feature_5": np.random.uniform(8000, 65535),    # Normal window size
            "feature_6": np.random.uniform(1000, 1460),     # Full-sized packets
            "feature_7": np.random.uniform(0.001, 0.01)     # Small inter-arrival time
        }
        samples.append(sample)
    
    # 3. Generate SYN flood DDoS attack
    for i in range(int(num_samples * 0.15)):
        sample = {
            "feature_0": np.random.uniform(0.001, 0.01),    # Very short flows
            "feature_1": np.random.uniform(40, 100),        # Small data size
            "feature_2": np.random.uniform(1, 3),           # Few packets
            "feature_3": np.random.uniform(5000, 15000),    # High byte rate
            "feature_4": np.random.uniform(30, 250),        # Variable TTL
            "feature_5": np.random.uniform(1000, 5000),     # Small window size
            "feature_6": np.random.uniform(40, 100),        # Small packet size (SYN)
            "feature_7": np.random.uniform(0.0001, 0.001)   # Very small inter-arrival time
        }
        samples.append(sample)
    
    # 4. Generate port scanning attack
    for i in range(int(num_samples * 0.15)):
        sample = {
            "feature_0": np.random.uniform(0.01, 0.1),      # Short flows
            "feature_1": np.random.uniform(50, 150),        # Small data size
            "feature_2": np.random.uniform(1, 5),           # Few packets
            "feature_3": np.random.uniform(500, 3000),      # Moderate byte rate
            "feature_4": np.random.uniform(30, 64),         # Normal TTL
            "feature_5": np.random.uniform(1000, 8000),     # Small window size
            "feature_6": np.random.uniform(40, 60),         # Small packet size
            "feature_7": np.random.uniform(0.005, 0.05)     # Small inter-arrival time
        }
        samples.append(sample)
    
    # 5. Generate HTTP flood attack
    for i in range(int(num_samples * 0.15)):
        sample = {
            "feature_0": np.random.uniform(0.1, 1.0),       # Medium flows
            "feature_1": np.random.uniform(1000, 5000),     # Normal HTTP request size
            "feature_2": np.random.uniform(10, 50),         # Normal packet count
            "feature_3": np.random.uniform(10000, 50000),   # Very high byte rate
            "feature_4": np.random.uniform(40, 64),         # Normal TTL
            "feature_5": np.random.uniform(5000, 65535),    # Normal window size
            "feature_6": np.random.uniform(200, 500),       # Normal HTTP packet size
            "feature_7": np.random.uniform(0.0001, 0.001)   # Very small inter-arrival time
        }
        samples.append(sample)
    
    # 6. Generate data exfiltration attack
    for i in range(int(num_samples * 0.1)):
        sample = {
            "feature_0": np.random.uniform(10.0, 60.0),     # Long flow duration
            "feature_1": np.random.uniform(10000000, 100000000), # Very large data transfer
            "feature_2": np.random.uniform(5000, 20000),    # Many packets
            "feature_3": np.random.uniform(1000, 5000),     # Moderate byte rate
            "feature_4": np.random.uniform(50, 64),         # Normal TTL
            "feature_5": np.random.uniform(8000, 65535),    # Normal window size
            "feature_6": np.random.uniform(1000, 1460),     # Full-sized packets
            "feature_7": np.random.uniform(0.001, 0.01)     # Small inter-arrival time
        }
        samples.append(sample)
    
    # 7. Generate brute force attack
    for i in range(int(num_samples * 0.15)):
        sample = {
            "feature_0": np.random.uniform(5.0, 20.0),      # Medium-long flow
            "feature_1": np.random.uniform(5000, 50000),    # Small-medium data size
            "feature_2": np.random.uniform(100, 500),       # Many packets
            "feature_3": np.random.uniform(500, 2000),      # Moderate byte rate
            "feature_4": np.random.uniform(50, 64),         # Normal TTL
            "feature_5": np.random.uniform(5000, 65535),    # Normal window size
            "feature_6": np.random.uniform(60, 200),        # Small packet size
            "feature_7": np.random.uniform(0.01, 0.1)       # Regular inter-arrival time
        }
        samples.append(sample)
    
    return samples

def visualize_results(results, samples):
    """
    Visualize prediction results.
    
    Args:
        results: List of prediction results
        samples: List of original samples
    """
    # Create figure for visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Plot risk score distribution
    plt.subplot(2, 2, 1)
    risk_scores = [r['risk_score'] for r in results]
    plt.hist(risk_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title('Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    
    # 2. Plot confidence vs risk score
    plt.subplot(2, 2, 2)
    confidences = [r['confidence'] for r in results]
    plt.scatter(risk_scores, confidences, alpha=0.6)
    plt.title('Confidence vs Risk Score')
    plt.xlabel('Risk Score')
    plt.ylabel('Confidence')
    
    # 3. Plot feature importance (using most important features from first attack)
    plt.subplot(2, 2, 3)
    attack_results = [r for r in results if r['is_attack']]
    if attack_results:
        explanation = predictor.explain(samples[results.index(attack_results[0])])
        if 'abnormal_features' in explanation and explanation['abnormal_features']:
            features = [f['name'] for f in explanation['abnormal_features']]
            importances = [f['importance'] for f in explanation['abnormal_features']]
            plt.barh(features, importances, color='salmon')
            plt.title('Feature Importance for Attack Detection')
            plt.xlabel('Importance')
        else:
            plt.text(0.5, 0.5, "No feature importance data", ha='center', va='center')
    else:
        plt.text(0.5, 0.5, "No attacks detected", ha='center', va='center')
    
    # 4. Plot attack type distribution
    plt.subplot(2, 2, 4)
    attack_types = []
    for i, r in enumerate(results):
        if r['is_attack']:
            explanation = predictor.explain(samples[i])
            if 'threat_intelligence' in explanation:
                attack_types.append(explanation['threat_intelligence']['attack_type'])
    
    if attack_types:
        from collections import Counter
        type_counts = Counter(attack_types)
        plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Attack Type Distribution')
    else:
        plt.text(0.5, 0.5, "No attacks detected", ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the visualization
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, "inference_results.png"))
    logging.info(f"Results visualization saved to {os.path.join(output_dir, 'inference_results.png')}")

def test_inference() -> None:
    """
    Test the inference engine with different sample types.
    """
    # Set the model path to the correct directory
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "models/multi_dataset/chained_transfer_improved/best_model.keras")
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "models/multi_dataset/chained_transfer_improved/scaler.pkl")
    
    # Create a predictor instance with all improvements
    logging.info("Creating NeuraShield predictor with optimal settings...")
    logging.info(f"Using model at: {model_path}")
    global predictor
    predictor = NeuraShieldPredictor(
        model_path=model_path, 
        scaler_path=scaler_path,
        threshold=0.1,  # Using optimal threshold from training
        calibrate_confidence=True,
        use_ensemble=True
    )
    
    # Generate advanced test samples
    logging.info("Generating advanced test samples with realistic patterns...")
    samples = generate_advanced_test_samples(50)
    
    # Test batch prediction for efficiency
    logging.info("Performing batch prediction on all samples...")
    start_time = time.time()
    results = predictor.predict_batch(samples)
    batch_time = time.time() - start_time
    
    # Calculate statistics
    attack_count = sum(1 for r in results if r['prediction'] == 'attack')
    benign_count = sum(1 for r in results if r['prediction'] == 'benign')
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_risk_score = sum(r['risk_score'] for r in results) / len(results)
    
    # Display summary
    logging.info("\n=== Prediction Summary ===")
    logging.info(f"Total samples processed: {len(results)}")
    logging.info(f"Attacks detected: {attack_count} ({attack_count/len(results)*100:.1f}%)")
    logging.info(f"Benign traffic: {benign_count} ({benign_count/len(results)*100:.1f}%)")
    logging.info(f"Average confidence: {avg_confidence:.4f}")
    logging.info(f"Average risk score: {avg_risk_score:.1f}")
    logging.info(f"Processing time: {batch_time:.2f}s ({batch_time*1000/len(samples):.2f}ms per sample)")
    
    # Test detailed explanation for first detected attack
    logging.info("\n=== Detailed Threat Analysis ===")
    attack_indices = [i for i, r in enumerate(results) if r['is_attack']]
    
    if attack_indices:
        first_attack_idx = attack_indices[0]
        sample = samples[first_attack_idx]
        
        logging.info(f"Analyzing attack sample #{first_attack_idx+1}:")
        
        # Get detailed explanation
        start_time = time.time()
        explanation = predictor.explain(sample)
        explain_time = time.time() - start_time
        
        # Print threat intelligence
        if "threat_intelligence" in explanation:
            ti = explanation["threat_intelligence"]
            logging.info(f"Attack type: {ti['attack_type']}")
            logging.info(f"Severity: {ti['severity']}")
            logging.info(f"Confidence: {ti['confidence']:.4f}")
            
            logging.info("\nKey indicators:")
            abnormal = explanation.get("abnormal_features", [])
            for feature in abnormal[:5]:  # Top 5 features
                logging.info(f"- {feature['name']}: {feature['value']:.4f} (importance: {feature['importance']:.2f})")
            
            logging.info("\nRecommended mitigations:")
            for i, m in enumerate(ti['mitigations']):
                logging.info(f"{i+1}. {m}")
        
        logging.info(f"\nExplanation completed in {explain_time*1000:.2f}ms")
    else:
        logging.info("No attacks detected in the test samples.")
    
    # Visualize the results
    try:
        logging.info("\n=== Generating Visualizations ===")
        visualize_results(results, samples)
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
    
    # Store results for testing different thresholds
    logging.info("\n=== Threshold Sensitivity Analysis ===")
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
    threshold_results = []
    
    for threshold in thresholds:
        # Create predictor with this threshold
        test_predictor = NeuraShieldPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            threshold=threshold,
            calibrate_confidence=True
        )
        
        # Test on all samples
        test_results = test_predictor.predict_batch(samples)
        attack_count = sum(1 for r in test_results if r['prediction'] == 'attack')
        
        threshold_results.append({
            "threshold": threshold,
            "attacks": attack_count,
            "benign": len(samples) - attack_count,
            "attack_percent": f"{attack_count/len(samples)*100:.1f}%"
        })
    
    # Display threshold sensitivity results
    logging.info("Effect of different classification thresholds:")
    headers = ["Threshold", "Attacks", "Benign", "Attack %"]
    table_data = [[r["threshold"], r["attacks"], r["benign"], r["attack_percent"]] for r in threshold_results]
    logging.info("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    logging.info("\nInference test completed successfully.")

def main() -> None:
    """
    Main function.
    """
    print("=== NeuraShield Advanced Inference Test ===\n")
    
    try:
        # Ensure output directories exist
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/multi_dataset/chained_transfer_improved"), exist_ok=True)
        
        # Run inference tests
        test_inference()
        print("\nTest completed successfully. See inference_results.png for visualizations.")
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print("\nTest failed. See log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 