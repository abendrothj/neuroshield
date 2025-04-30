#!/usr/bin/env python3

"""
Mock API server for testing the threat detection daemon
Provides network data and simulates the blockchain endpoint
"""

from flask import Flask, jsonify, request
import random
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mock_api_server')

app = Flask(__name__)

@app.route('/api/v1/network-data', methods=['GET'])
def get_network_data():
    """Generate random network traffic data"""
    logger.info("Received request for network data")
    
    data = []
    for i in range(3):
        # 20% chance of generating suspicious traffic
        is_suspicious = random.random() < 0.2
        
        # Generate more suspicious-looking values if is_suspicious is True
        bytes_sent = random.randint(100, 10000)
        bytes_received = random.randint(100, 10000)
        packets_sent = random.randint(1, 100) 
        packets_received = random.randint(1, 100)
        duration_ms = random.randint(10, 1000)
        
        if is_suspicious:
            # Suspicious traffic has anomalous patterns
            if random.random() < 0.5:
                # High outbound traffic
                bytes_sent *= 20
                packets_sent *= 5
            else:
                # High inbound traffic
                bytes_received *= 20
                packets_received *= 5
                duration_ms *= 3
        
        record = {
            'timestamp': time.time(),
            'source_ip': f"192.168.1.{random.randint(1, 254)}",
            'source_port': random.randint(10000, 65535),
            'destination_ip': f"10.0.0.{random.randint(1, 254)}",
            'destination_port': random.choice([80, 443, 22, 25, 53]),
            'protocol': random.choice(['TCP', 'UDP']),
            'bytes_sent': bytes_sent,
            'bytes_received': bytes_received,
            'packets_sent': packets_sent,
            'packets_received': packets_received,
            'duration_ms': duration_ms
        }
        data.append(record)
    
    logger.info(f"Returning {len(data)} network traffic records")
    return jsonify(data)

@app.route('/api/v1/events', methods=['POST'])
def receive_event():
    """Mock blockchain endpoint"""
    logger.info("Received blockchain event")
    
    # Log the received data
    data = request.json
    logger.info(f"Event data: {data}")
    
    # Generate a response
    event_id = f"test-event-{int(time.time())}"
    logger.info(f"Generated event ID: {event_id}")
    
    return jsonify({
        "status": "success", 
        "eventId": event_id,
        "timestamp": time.time()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == '__main__':
    logger.info("Starting mock API server on http://localhost:5000")
    app.run(host='localhost', port=5000) 