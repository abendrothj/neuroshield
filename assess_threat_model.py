#!/usr/bin/env python3

"""
NeuraShield Threat Model Assessment Tool
This script combines analysis and visualization tools for comprehensive model assessment
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def run_analysis(model_path, data_path, output_dir, preprocessed=False, scaler_path=None):
    """Run the analysis tool on the model"""
    try:
        from analyze_model import analyze_model
        
        logging.info("Starting model analysis...")
        
        # Create output directory
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Run analysis
        success = analyze_model(
            model_path=model_path,
            data_path=data_path,
            output_dir=analysis_dir,
            preprocessed=preprocessed,
            scaler_path=scaler_path
        )
        
        if success:
            logging.info(f"Analysis completed successfully. Results saved to {analysis_dir}")
            return analysis_dir
        else:
            logging.error("Analysis failed.")
            return None
            
    except ImportError:
        # Fall back to subprocess if direct import doesn't work
        cmd = [
            sys.executable, "analyze_model.py",
            "--model-path", model_path,
            "--data-path", data_path,
            "--output-dir", os.path.join(output_dir, "analysis")
        ]
        
        if preprocessed:
            cmd.append("--preprocessed")
            
        if scaler_path:
            cmd.extend(["--scaler-path", scaler_path])
        
        logging.info(f"Running analysis as subprocess: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Analysis completed successfully.\n{result.stdout}")
            return os.path.join(output_dir, "analysis")
        else:
            logging.error(f"Analysis failed with code {result.returncode}.\n{result.stderr}")
            return None

def run_visualization(model_path, data_path, output_dir, feature_names=None, class_names=None, 
                      preprocessed=False, scaler_path=None):
    """Run the visualization tool on the model"""
    try:
        from visualize_model import visualize_model_performance, load_model, load_data
        
        logging.info("Starting model visualization...")
        
        # Create output directory
        visualization_dir = os.path.join(output_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Load model
        model = load_model(model_path)
        if model is None:
            logging.error("Failed to load model for visualization.")
            return None
        
        # Load data
        X_test, y_test = load_data(
            data_path,
            preprocessed=preprocessed,
            scaler_path=scaler_path
        )
        if X_test is None or y_test is None:
            logging.error("Failed to load data for visualization.")
            return None
        
        # Run visualization
        visualize_model_performance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            output_dir=visualization_dir,
            class_names=class_names
        )
        
        logging.info(f"Visualization completed successfully. Results saved to {visualization_dir}")
        return visualization_dir
            
    except ImportError:
        # Fall back to subprocess if direct import doesn't work
        cmd = [
            sys.executable, "visualize_model.py",
            "--model-path", model_path,
            "--data-path", data_path,
            "--output-dir", os.path.join(output_dir, "visualization")
        ]
        
        if preprocessed:
            cmd.append("--preprocessed")
            
        if scaler_path:
            cmd.extend(["--scaler-path", scaler_path])
            
        if feature_names:
            cmd.extend(["--feature-names", feature_names])
            
        if class_names:
            cmd.extend(["--class-names", class_names])
        
        logging.info(f"Running visualization as subprocess: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Visualization completed successfully.\n{result.stdout}")
            return os.path.join(output_dir, "visualization")
        else:
            logging.error(f"Visualization failed with code {result.returncode}.\n{result.stderr}")
            return None

def generate_summary_report(analysis_dir, visualization_dir, output_dir):
    """Generate a summary report linking to both analysis and visualization results"""
    try:
        summary_path = os.path.join(output_dir, "assessment_summary.html")
        
        # Check if we have results to report
        has_analysis = analysis_dir is not None and os.path.exists(analysis_dir)
        has_visualization = visualization_dir is not None and os.path.exists(visualization_dir)
        
        # Find report files
        analysis_report = os.path.join(analysis_dir, "analysis_report.html") if has_analysis else None
        visualization_dashboard = os.path.join(visualization_dir, "performance_dashboard.png") if has_visualization else None
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuraShield Model Assessment Summary</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                .report-link {{
                    display: inline-block;
                    margin: 10px 0;
                    padding: 10px 20px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                }}
                .report-link:hover {{
                    background-color: #2980b9;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NeuraShield Model Assessment Summary</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="section">
                    <h2>Analysis Results</h2>
                    {"<p>Detailed analysis was performed to evaluate the model's architecture, performance, and error patterns.</p>" if has_analysis else "<p>No analysis results available.</p>"}
                    {f'<a class="report-link" href="{os.path.relpath(analysis_report, output_dir)}" target="_blank">View Full Analysis Report</a>' if analysis_report and os.path.exists(analysis_report) else ""}
                </div>
                
                <div class="section">
                    <h2>Visualization Results</h2>
                    {"<p>Comprehensive visualizations were generated to illustrate the model's performance and behavior.</p>" if has_visualization else "<p>No visualization results available.</p>"}
                    {f'<div class="image-container"><img src="{os.path.relpath(visualization_dashboard, output_dir)}" alt="Performance Dashboard"></div>' if visualization_dashboard and os.path.exists(visualization_dashboard) else ""}
                    {f'<a class="report-link" href="{os.path.relpath(visualization_dir, output_dir)}" target="_blank">View All Visualizations</a>' if has_visualization else ""}
                </div>
                
                <div class="section">
                    <h2>Next Steps</h2>
                    <p>Based on these assessment results, consider the following actions:</p>
                    <ul>
                        <li>Review the model's performance metrics to identify strengths and weaknesses</li>
                        <li>Examine error patterns to understand which types of threats are most challenging to detect</li>
                        <li>Use the feature importance analysis to optimize feature selection</li>
                        <li>Consider fine-tuning the model using the `finetune_model.py` script if performance needs improvement</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(summary_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Summary report generated and saved to {summary_path}")
        return summary_path
    except Exception as e:
        logging.error(f"Error generating summary report: {str(e)}")
        return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Assess threat detection model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model to assess')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to test data for assessment')
    parser.add_argument('--output-dir', type=str, default='model_assessment',
                        help='Directory to save assessment results')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Indicate if the data is already preprocessed')
    parser.add_argument('--scaler-path', type=str,
                        help='Path to saved scaler for data preprocessing')
    parser.add_argument('--feature-names', type=str,
                        help='Path to JSON file with feature names')
    parser.add_argument('--class-names', type=str,
                        help='Path to JSON file with class names')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip the analysis step')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip the visualization step')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load feature names and class names if provided
    feature_names = None
    class_names = None
    
    if args.feature_names:
        try:
            import json
            with open(args.feature_names, 'r') as f:
                feature_names = json.load(f)
        except Exception as e:
            logging.warning(f"Error loading feature names: {str(e)}")
    
    if args.class_names:
        try:
            import json
            with open(args.class_names, 'r') as f:
                class_names = json.load(f)
        except Exception as e:
            logging.warning(f"Error loading class names: {str(e)}")
    
    # Run assessment steps
    analysis_dir = None
    visualization_dir = None
    
    if not args.skip_analysis:
        analysis_dir = run_analysis(
            args.model_path,
            args.data_path,
            output_dir,
            preprocessed=args.preprocessed,
            scaler_path=args.scaler_path
        )
    
    if not args.skip_visualization:
        visualization_dir = run_visualization(
            args.model_path,
            args.data_path,
            output_dir,
            feature_names=feature_names,
            class_names=class_names,
            preprocessed=args.preprocessed,
            scaler_path=args.scaler_path
        )
    
    # Generate summary report
    summary_report = generate_summary_report(analysis_dir, visualization_dir, output_dir)
    
    if summary_report:
        logging.info(f"Assessment complete. Summary report saved to {summary_report}")
        return 0
    else:
        logging.error("Assessment failed to generate a summary report.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 