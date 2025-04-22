#!/usr/bin/env python3

"""
NeuraShield AI Model Analysis Script
This script analyzes the performance of a trained threat detection model
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ThreatDetectionSystem
from threat_detection_system import ThreatDetectionSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="NeuraShield AI Model Analysis")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data (if available)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic test data if no test data is provided"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of synthetic test samples to generate (if --generate-data is used)"
    )
    parser.add_argument(
        "--plot-heatmap",
        action="store_true",
        help="Generate feature importance heatmap"
    )
    
    return parser.parse_args()

def find_model_path(model_path):
    """Find the most recent model if no path specified"""
    if model_path is not None and os.path.exists(model_path):
        return model_path
        
    # Check the models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        models_dir = "ai_models/models"
        if not os.path.exists(models_dir):
            logging.error("No models directory found")
            return None
    
    # Find the most recent model directory
    model_dirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("models_")]
    
    if not model_dirs:
        logging.error("No model directories found")
        return None
    
    # Sort by creation time (newest first)
    model_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    # Get the newest model directory
    newest_model_dir = model_dirs[0]
    model_path = os.path.join(newest_model_dir, "threat_detection_model")
    
    if os.path.exists(model_path):
        logging.info(f"Using most recent model: {model_path}")
        return model_path
    else:
        logging.error(f"Model not found in {model_path}")
        return None

def load_or_generate_data(args, system):
    """Load test data or generate synthetic data"""
    if args.test_data is not None and os.path.exists(args.test_data):
        # Load test data from file
        logging.info(f"Loading test data from {args.test_data}")
        data = np.load(args.test_data)
        X_test = data["X"]
        y_test = data["y"]
    elif args.generate_data:
        # Generate synthetic test data
        logging.info(f"Generating {args.samples} synthetic test samples")
        try:
            # Try to load the dataset from the UNSW_NB15 directory
            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "UNSW_NB15")
            
            if os.path.exists(dataset_path):
                logging.info(f"Loading dataset from {dataset_path}")
                X, y = system.load_unsw_dataset(dataset_path)
                # Split into train and test sets
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                # If dataset doesn't exist, generate synthetic data
                logging.info("Generating synthetic data")
                X, y = system.generate_training_data(n_samples=args.samples)
                X_test, y_test = X, y
        except Exception as e:
            logging.error(f"Error loading/generating data: {str(e)}")
            # Fallback to completely synthetic data
            logging.info("Falling back to synthetic data generation")
            X_test = np.random.rand(args.samples, system.model.input_shape[0])
            y_test = np.random.randint(0, len(system.threat_types), size=args.samples)
    else:
        logging.error("No test data provided and --generate-data not specified")
        return None, None
    
    logging.info(f"Test data shape: X: {X_test.shape}, y: {y_test.shape}")
    return X_test, y_test

def analyze_model_performance(X_test, y_test, system, output_dir):
    """Analyze model performance on test data"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate predictions
    try:
        y_pred_proba = system.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Create classification report
        class_names = [system.threat_types.get(i, f"Unknown ({i})") for i in range(len(system.threat_types))]
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"\nClassification Report:\n{report}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{timestamp}.png"))
        plt.close()
        
        # Generate ROC curves for each class
        plt.figure(figsize=(10, 8))
        for i in range(len(system.threat_types)):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{system.threat_types.get(i, f"Class {i}")}: (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"roc_curves_{timestamp}.png"))
        plt.close()
        
        # Generate precision-recall curves for each class
        plt.figure(figsize=(10, 8))
        for i in range(len(system.threat_types)):
            precision, recall, _ = precision_recall_curve((y_test == i).astype(int), y_pred_proba[:, i])
            plt.plot(recall, precision, lw=2, label=f'{system.threat_types.get(i, f"Class {i}")}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"precision_recall_curves_{timestamp}.png"))
        plt.close()
        
        # Create class distribution plot
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=y_test, palette='viridis')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        plt.title('Class Distribution in Test Data')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_distribution_{timestamp}.png"))
        plt.close()
        
        # Test critical scenarios
        critical_scenarios = test_critical_scenarios(system, output_dir, timestamp)
        
        # Save analysis results
        analysis_results = {
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "timestamp": timestamp,
            "critical_scenarios": critical_scenarios
        }
        
        with open(os.path.join(output_dir, f"analysis_results_{timestamp}.json"), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logging.info(f"Analysis results saved to {os.path.join(output_dir, f'analysis_results_{timestamp}.json')}")
        
        return analysis_results
        
    except Exception as e:
        logging.error(f"Error analyzing model performance: {str(e)}")
        raise

def test_critical_scenarios(system, output_dir, timestamp):
    """Test model performance on critical scenarios"""
    logging.info("Testing critical scenarios...")
    
    # Define critical scenarios
    scenarios = [
        {
            "name": "High-Volume DDoS",
            "description": "High-volume DDoS attack with extreme network metrics",
            "features": np.random.normal(0.9, 0.05, system.model.input_shape[0])
        },
        {
            "name": "Low-and-Slow DDoS",
            "description": "Low-and-slow DDoS attack that's harder to detect",
            "features": np.random.normal(0.6, 0.1, system.model.input_shape[0])
        },
        {
            "name": "Advanced Persistent Threat",
            "description": "Sophisticated APT with stealthy behavior",
            "features": np.random.normal(0.5, 0.15, system.model.input_shape[0])
        },
        {
            "name": "Zero-Day Exploit",
            "description": "Unknown attack pattern (zero-day exploit)",
            "features": np.random.normal(0.7, 0.25, system.model.input_shape[0])
        },
        {
            "name": "Borderline Normal/Abnormal",
            "description": "Traffic at the boundary between normal and abnormal",
            "features": np.random.normal(0.4, 0.1, system.model.input_shape[0])
        }
    ]
    
    # Ensure all features are in range [0, 1]
    for scenario in scenarios:
        scenario["features"] = np.clip(scenario["features"], 0, 1)
    
    # Test each scenario
    scenario_results = []
    for scenario in scenarios:
        result = system.analyze_threat(scenario["features"])
        
        # Add scenario and result details
        scenario_data = {
            "name": scenario["name"],
            "description": scenario["description"],
            "prediction": result["prediction"],
            "is_threat": result["is_threat"],
            "confidence": float(result["confidence"]),
            "confidence_values": {
                system.threat_types[i]: float(result["confidence_values"][i]) 
                for i in range(len(result["confidence_values"]))
            }
        }
        
        logging.info(f"Scenario: {scenario['name']}")
        logging.info(f"  Prediction: {result['prediction']}")
        logging.info(f"  Is Threat: {result['is_threat']}")
        logging.info(f"  Confidence: {result['confidence']:.2f}%")
        
        scenario_results.append(scenario_data)
    
    # Create bar chart for scenario confidences
    plt.figure(figsize=(12, 8))
    scenario_names = [s["name"] for s in scenario_results]
    confidences = [s["confidence"] for s in scenario_results]
    is_threat = [s["is_threat"] for s in scenario_results]
    
    colors = ['red' if threat else 'green' for threat in is_threat]
    
    ax = plt.bar(scenario_names, confidences, color=colors)
    plt.title('Threat Confidence for Critical Scenarios')
    plt.xlabel('Scenario')
    plt.ylabel('Confidence (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"critical_scenarios_{timestamp}.png"))
    plt.close()
    
    # Save scenario results
    with open(os.path.join(output_dir, f"critical_scenarios_{timestamp}.json"), 'w') as f:
        json.dump(scenario_results, f, indent=2)
    
    return scenario_results

def analyze_feature_importance(X_test, y_test, system, output_dir, timestamp):
    """Analyze feature importance using permutation importance"""
    if not hasattr(system.model, 'input_shape'):
        logging.warning("Model does not have input_shape attribute. Skipping feature importance analysis.")
        return
    
    logging.info("Analyzing feature importance...")
    
    # Get baseline accuracy
    baseline_proba = system.model.predict(X_test)
    baseline_pred = np.argmax(baseline_proba, axis=1)
    baseline_accuracy = np.mean(baseline_pred == y_test)
    
    # Calculate permutation importance
    feature_importance = []
    n_features = X_test.shape[1]
    for i in range(n_features):
        # Make a copy of the test data
        X_permuted = X_test.copy()
        
        # Permute this feature
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # Make predictions and calculate accuracy
        permuted_proba = system.model.predict(X_permuted)
        permuted_pred = np.argmax(permuted_proba, axis=1)
        permuted_accuracy = np.mean(permuted_pred == y_test)
        
        # Calculate importance as decrease in accuracy
        importance = baseline_accuracy - permuted_accuracy
        feature_importance.append({
            "feature_index": i,
            "importance": float(importance)
        })
        
        logging.info(f"Feature {i} importance: {importance:.5f}")
    
    # Sort features by importance
    feature_importance.sort(key=lambda x: abs(x["importance"]), reverse=True)
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    features = [f"Feature {f['feature_index']}" for f in feature_importance[:20]]  # Top 20 features
    importances = [f["importance"] for f in feature_importance[:20]]
    
    colors = ['red' if imp < 0 else 'blue' for imp in importances]
    
    plt.bar(features, importances, color=colors)
    plt.title('Feature Importance (Permutation Method)')
    plt.xlabel('Feature')
    plt.ylabel('Importance (Accuracy Change)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{timestamp}.png"))
    plt.close()
    
    # Save feature importance results
    with open(os.path.join(output_dir, f"feature_importance_{timestamp}.json"), 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    return feature_importance

def generate_feature_heatmap(X_test, y_test, system, output_dir, timestamp):
    """Generate feature correlation heatmap"""
    if X_test.shape[1] > 100:
        logging.warning("Too many features for heatmap. Skipping.")
        return
    
    logging.info("Generating feature correlation heatmap...")
    
    # Combine features with targets for correlation analysis
    data = np.column_stack([X_test, y_test])
    cols = [f"Feature_{i}" for i in range(X_test.shape[1])] + ["Target"]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)
    
    # Generate heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                xticklabels=cols, yticklabels=cols)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_correlation_{timestamp}.png"))
    plt.close()
    
    # Generate target-feature correlation plot
    if X_test.shape[1] <= 30:  # Only if number of features is reasonable
        target_correlations = corr_matrix[-1, :-1]  # Correlations between target and features
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(target_correlations)), target_correlations, 
                color=['red' if c < 0 else 'blue' for c in target_correlations])
        plt.xticks(range(len(target_correlations)), [f"F{i}" for i in range(len(target_correlations))], 
                 rotation=90)
        plt.title('Feature-Target Correlation')
        plt.xlabel('Features')
        plt.ylabel('Correlation with Target')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_target_correlation_{timestamp}.png"))
        plt.close()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find model path
    model_path = find_model_path(args.model_path)
    if model_path is None:
        return
    
    # Load the model
    system = ThreatDetectionSystem(model_path=model_path)
    try:
        system.load_model()
        logging.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return
    
    # Get model information
    model_info = system.model.get_model_info()
    logging.info(f"Model version: {model_info['model_version']}")
    logging.info(f"Input shape: {model_info['input_shape']}")
    logging.info(f"Number of classes: {model_info['num_classes']}")
    logging.info(f"Number of layers: {model_info['layers']}")
    logging.info(f"Trainable parameters: {model_info['trainable_params']}")
    
    # Load or generate test data
    X_test, y_test = load_or_generate_data(args, system)
    if X_test is None or y_test is None:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Analyze model performance
    analysis_results = analyze_model_performance(X_test, y_test, system, args.output_dir)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(X_test, y_test, system, args.output_dir, timestamp)
    
    # Generate feature heatmap if requested
    if args.plot_heatmap:
        generate_feature_heatmap(X_test, y_test, system, args.output_dir, timestamp)
    
    logging.info(f"Analysis completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 