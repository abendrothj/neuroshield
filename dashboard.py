#!/usr/bin/env python3

"""
NeuraShield Dashboard
A Streamlit-based dashboard for the NeuraShield threat detection system.
"""

import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any, Tuple
import threading
import random

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/dashboard.log"))
    ]
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"

# Class for managing dashboard data
class DashboardData:
    """Class for managing dashboard data"""
    
    def __init__(self, api_url: str = DEFAULT_API_URL, max_history: int = 1000):
        """
        Initialize dashboard data.
        
        Args:
            api_url: URL of the NeuraShield API
            max_history: Maximum number of predictions to keep in history
        """
        self.api_url = api_url
        self.max_history = max_history
        self.predictions = []
        self.alerts = []
        self.api_health = {"status": "unknown", "model_loaded": False, "version": "unknown"}
        self.last_update = datetime.now()
        self.mock_mode = False
        
        # Load prediction logs if available
        self._load_prediction_logs()
    
    def _load_prediction_logs(self):
        """Load prediction logs from file"""
        try:
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/predictions.jsonl")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-self.max_history:]:
                        prediction = json.loads(line.strip())
                        self.predictions.append(prediction)
                        
                        # Add to alerts if it's an attack
                        if prediction.get("prediction") == "attack" and prediction.get("probability", 0) > 0.7:
                            self.alerts.append(prediction)
                
                logging.info(f"Loaded {len(self.predictions)} predictions from logs")
        except Exception as e:
            logging.error(f"Error loading prediction logs: {str(e)}")
    
    def check_api_health(self) -> bool:
        """
        Check API health.
        
        Returns:
            True if API is healthy, False otherwise
        """
        if self.mock_mode:
            self.api_health = {"status": "healthy", "model_loaded": True, "version": "1.0.0"}
            return True
            
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            if response.status_code == 200:
                self.api_health = response.json()
                return True
            else:
                self.api_health = {"status": "error", "model_loaded": False, "version": "unknown"}
                return False
        except Exception as e:
            logging.error(f"Error checking API health: {str(e)}")
            self.api_health = {"status": "error", "model_loaded": False, "version": "unknown"}
            return False
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent predictions.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of recent predictions
        """
        return self.predictions[-limit:]
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """
        Get prediction summary.
        
        Returns:
            Dictionary with prediction summary statistics
        """
        if not self.predictions:
            return {
                "total": 0,
                "attack": 0,
                "benign": 0,
                "attack_percentage": 0,
                "avg_confidence": 0
            }
        
        total = len(self.predictions)
        attack_count = sum(1 for p in self.predictions if p.get("prediction") == "attack")
        benign_count = total - attack_count
        
        # Calculate average confidence
        confidence_sum = sum(p.get("confidence", 0) for p in self.predictions)
        avg_confidence = confidence_sum / total if total > 0 else 0
        
        return {
            "total": total,
            "attack": attack_count,
            "benign": benign_count,
            "attack_percentage": (attack_count / total) * 100 if total > 0 else 0,
            "avg_confidence": avg_confidence
        }
    
    def get_predictions_by_time(self, hours: int = 24) -> pd.DataFrame:
        """
        Get predictions grouped by time.
        
        Args:
            hours: Number of hours to include in the time series
            
        Returns:
            DataFrame with predictions grouped by time
        """
        if not self.predictions:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["timestamp", "attack_count", "benign_count", "total"])
        
        # Convert predictions to DataFrame
        df = pd.DataFrame(self.predictions)
        
        # Convert timestamp string to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        df = df[df["timestamp"] >= cutoff_time]
        
        # Resample to hourly buckets
        df["is_attack"] = df["prediction"] == "attack"
        df["is_benign"] = df["prediction"] == "benign"
        
        # Group by hour
        hourly = df.set_index("timestamp").resample("1H").agg({
            "is_attack": "sum",
            "is_benign": "sum"
        }).reset_index()
        
        # Rename columns
        hourly = hourly.rename(columns={
            "is_attack": "attack_count",
            "is_benign": "benign_count"
        })
        
        # Add total column
        hourly["total"] = hourly["attack_count"] + hourly["benign_count"]
        
        return hourly
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.alerts[-limit:]
    
    def add_prediction(self, prediction: Dict[str, Any]):
        """
        Add a prediction to the history.
        
        Args:
            prediction: Prediction dictionary
        """
        self.predictions.append(prediction)
        
        # Trim predictions if necessary
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history:]
        
        # Add to alerts if it's an attack with high confidence
        if prediction.get("prediction") == "attack" and prediction.get("probability", 0) > 0.7:
            self.alerts.append(prediction)
            
            # Trim alerts if necessary
            if len(self.alerts) > self.max_history:
                self.alerts = self.alerts[-self.max_history:]
        
        self.last_update = datetime.now()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on historical predictions.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.predictions or len(self.predictions) < 10:
            # Return mock data if not enough predictions
            features = [f"feature_{i}" for i in range(8)]
            importance = [random.uniform(0.5, 1.0) for _ in range(8)]
            return pd.DataFrame({"feature": features, "importance": importance})
        
        # Get feature values from predictions
        feature_values = []
        for pred in self.predictions:
            if "feature_values" in pred:
                feature_values.append(pred["feature_values"])
        
        if not feature_values:
            # Return mock data if no feature values
            features = [f"feature_{i}" for i in range(8)]
            importance = [random.uniform(0.5, 1.0) for _ in range(8)]
            return pd.DataFrame({"feature": features, "importance": importance})
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_values)
        
        # Add prediction column
        df["is_attack"] = [p.get("prediction") == "attack" for p in self.predictions]
        
        # Calculate correlation with prediction
        correlations = {}
        for col in df.columns:
            if col != "is_attack":
                corr = df[col].corr(df["is_attack"])
                # Convert NaN to 0
                if pd.isna(corr):
                    corr = 0
                correlations[col] = abs(corr)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            "feature": list(correlations.keys()),
            "importance": list(correlations.values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        return importance_df
    
    def simulate_prediction(self):
        """Simulate a prediction for demo purposes"""
        if random.random() < 0.2:  # 20% chance of attack
            prediction = "attack"
            probability = random.uniform(0.7, 0.99)
        else:
            prediction = "benign"
            probability = random.uniform(0.01, 0.3)
        
        # Generate feature values
        feature_values = {}
        for i in range(8):
            feature_values[f"feature_{i}"] = random.uniform(0, 1)
        
        # Create prediction dictionary
        pred = {
            "prediction": prediction,
            "probability": probability,
            "confidence": probability if prediction == "attack" else 1 - probability,
            "threshold": 0.5,
            "timestamp": datetime.now().isoformat(),
            "feature_values": feature_values,
            "traffic_data": {
                "src_ip": f"192.168.1.{random.randint(1, 254)}",
                "dst_ip": f"10.0.0.{random.randint(1, 254)}",
                "src_port": random.randint(1024, 65535),
                "dst_port": random.choice([80, 443, 22, 25, 53]),
                "protocol": random.choice(["TCP", "UDP"]),
                "packet_count": random.randint(10, 1000),
                "byte_count": random.randint(1000, 100000)
            }
        }
        
        self.add_prediction(pred)
        
        return pred

# Dashboard UI
def create_dashboard():
    """Create the Streamlit dashboard"""
    # Set page title and favicon
    st.set_page_config(
        page_title="NeuraShield Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create dashboard data instance
    if "dashboard_data" not in st.session_state:
        st.session_state.dashboard_data = DashboardData()
    
    dashboard_data = st.session_state.dashboard_data
    
    # Sidebar
    with st.sidebar:
        st.title("NeuraShield Dashboard")
        
        st.subheader("Settings")
        
        # API settings
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        if api_url != dashboard_data.api_url:
            dashboard_data.api_url = api_url
        
        # Mock mode toggle
        mock_mode = st.checkbox("Demo Mode (no API connection)", value=dashboard_data.mock_mode)
        if mock_mode != dashboard_data.mock_mode:
            dashboard_data.mock_mode = mock_mode
        
        # Check API health
        api_status = dashboard_data.check_api_health()
        status_color = "green" if api_status else "red"
        st.markdown(f"<h4>API Status: <span style='color:{status_color}'>{dashboard_data.api_health['status']}</span></h4>", unsafe_allow_html=True)
        
        # Show model status
        model_status = "Loaded" if dashboard_data.api_health.get("model_loaded", False) else "Not Loaded"
        model_color = "green" if dashboard_data.api_health.get("model_loaded", False) else "red"
        st.markdown(f"<h4>Model Status: <span style='color:{model_color}'>{model_status}</span></h4>", unsafe_allow_html=True)
        
        # Version info
        st.markdown(f"API Version: {dashboard_data.api_health.get('version', 'unknown')}")
        
        # Last update time
        st.markdown(f"Last Update: {dashboard_data.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Simulate button (only in mock mode)
        if dashboard_data.mock_mode:
            if st.button("Simulate Prediction"):
                dashboard_data.simulate_prediction()
    
    # Main content
    # Top row with summary metrics
    st.subheader("Threat Detection Summary")
    summary = dashboard_data.get_prediction_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Traffic Analyzed", summary["total"])
    
    with col2:
        st.metric("Detected Attacks", summary["attack"])
    
    with col3:
        st.metric("Benign Traffic", summary["benign"])
    
    with col4:
        st.metric("Attack Percentage", f"{summary['attack_percentage']:.1f}%")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Detection Trends", "Recent Alerts", "Feature Analysis"])
    
    # Tab 1: Detection Trends
    with tab1:
        st.subheader("Detection Trends")
        
        # Get time series data
        hourly_data = dashboard_data.get_predictions_by_time(hours=24)
        
        if not hourly_data.empty:
            # Create attack trend chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add attack count bar chart
            fig.add_trace(
                go.Bar(
                    x=hourly_data["timestamp"],
                    y=hourly_data["attack_count"],
                    name="Attacks",
                    marker_color="crimson"
                ),
                secondary_y=False
            )
            
            # Add total traffic line chart
            fig.add_trace(
                go.Scatter(
                    x=hourly_data["timestamp"],
                    y=hourly_data["total"],
                    name="Total Traffic",
                    marker_color="blue",
                    mode="lines"
                ),
                secondary_y=True
            )
            
            # Add benign traffic bar chart
            fig.add_trace(
                go.Bar(
                    x=hourly_data["timestamp"],
                    y=hourly_data["benign_count"],
                    name="Benign",
                    marker_color="green"
                ),
                secondary_y=False
            )
            
            # Set titles
            fig.update_layout(
                title_text="Threat Detection History (24 hours)",
                barmode="stack"
            )
            
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Count", secondary_y=False)
            fig.update_yaxes(title_text="Total Traffic", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time series data available yet. Start analyzing traffic to see trends.")
        
        # Recent predictions table
        st.subheader("Recent Predictions")
        recent_predictions = dashboard_data.get_recent_predictions(limit=10)
        
        if recent_predictions:
            # Convert to DataFrame for display
            predictions_df = pd.DataFrame([
                {
                    "Time": p.get("timestamp", ""),
                    "Prediction": p.get("prediction", "unknown"),
                    "Confidence": f"{p.get('confidence', 0) * 100:.1f}%",
                    "Source IP": p.get("traffic_data", {}).get("src_ip", ""),
                    "Destination IP": p.get("traffic_data", {}).get("dst_ip", ""),
                    "Protocol": p.get("traffic_data", {}).get("protocol", "")
                }
                for p in recent_predictions
            ])
            
            # Style the DataFrame
            st.dataframe(
                predictions_df,
                column_config={
                    "Prediction": st.column_config.TextColumn(
                        "Prediction",
                        help="Prediction result",
                        width="medium"
                    ),
                    "Confidence": st.column_config.TextColumn(
                        "Confidence",
                        help="Prediction confidence",
                        width="small"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No predictions available yet.")
    
    # Tab 2: Recent Alerts
    with tab2:
        st.subheader("Recent Alerts")
        
        alerts = dashboard_data.get_alerts(limit=20)
        
        if alerts:
            # Convert to DataFrame for display
            alerts_df = pd.DataFrame([
                {
                    "Time": a.get("timestamp", ""),
                    "Confidence": f"{a.get('confidence', 0) * 100:.1f}%",
                    "Source IP": a.get("traffic_data", {}).get("src_ip", ""),
                    "Destination IP": a.get("traffic_data", {}).get("dst_ip", ""),
                    "Protocol": a.get("traffic_data", {}).get("protocol", ""),
                    "Packet Count": a.get("traffic_data", {}).get("packet_count", ""),
                    "Byte Count": a.get("traffic_data", {}).get("byte_count", "")
                }
                for a in alerts
            ])
            
            # Style the DataFrame
            st.dataframe(
                alerts_df,
                column_config={
                    "Confidence": st.column_config.TextColumn(
                        "Confidence",
                        help="Prediction confidence",
                        width="small"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Display alert details
            st.subheader("Alert Details")
            if alerts:
                alert_index = st.selectbox("Select alert to view details", range(len(alerts)), format_func=lambda i: f"{alerts[i].get('timestamp', '')} - {alerts[i].get('traffic_data', {}).get('src_ip', '')}")
                selected_alert = alerts[alert_index]
                
                # Show alert details
                st.json(selected_alert)
        else:
            st.info("No alerts available yet.")
    
    # Tab 3: Feature Analysis
    with tab3:
        st.subheader("Feature Importance")
        
        # Get feature importance
        importance_df = dashboard_data.get_feature_importance()
        
        if not importance_df.empty:
            # Create feature importance chart
            fig = px.bar(
                importance_df,
                x="feature",
                y="importance",
                title="Feature Importance for Attack Detection",
                labels={"feature": "Feature", "importance": "Importance Score"},
                color="importance",
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature descriptions
            st.subheader("Feature Descriptions")
            feature_descriptions = {
                "feature_0": "Flow duration (seconds)",
                "feature_1": "Total bytes transferred",
                "feature_2": "Total packet count",
                "feature_3": "Byte rate (bytes/sec)",
                "feature_4": "Average TTL value",
                "feature_5": "Average TCP window size",
                "feature_6": "Average packet size",
                "feature_7": "Average inter-arrival time"
            }
            
            # Display feature descriptions
            for feature, description in feature_descriptions.items():
                st.markdown(f"**{feature}**: {description}")
        else:
            st.info("Feature importance data not available yet.")
    
    # Simulate data in mock mode
    if dashboard_data.mock_mode:
        # Auto-generate a prediction every few seconds
        if random.random() < 0.3:  # 30% chance to generate a prediction on each refresh
            dashboard_data.simulate_prediction()

# Main function
def main():
    try:
        create_dashboard()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.error(f"Dashboard error: {str(e)}")

if __name__ == "__main__":
    main() 