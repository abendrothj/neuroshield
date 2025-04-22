#!/usr/bin/env python3

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("\n===== Testing Health Endpoint =====")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Model version: {data['model_version']}")
        print(f"  GPU available: {data['gpu_available']}")
        return True
    return False

def test_metrics():
    """Test the metrics endpoint"""
    print("\n===== Testing Metrics Endpoint =====")
    response = requests.get(f"{BASE_URL}/api/metrics")
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Model name: {data['model']['name']}")
        print(f"  Model version: {data['model']['version']}")
        print(f"  Model accuracy: {data['model']['accuracy']}")
        print(f"  GPU available: {data['hardware']['gpu_available']}")
        print(f"  Prediction count: {data['performance']['prediction_count']}")
        return True
    return False

def test_analyze(samples=1):
    """Test the analyze endpoint"""
    print(f"\n===== Testing Analyze Endpoint (with {samples} samples) =====")
    
    # Create test payload with specified number of samples
    payload = {"data": []}
    for i in range(samples):
        # Create a sample with 39 features
        sample = {f"feature_{j}": round(0.1 * ((i + j) % 10), 2) for j in range(39)}
        payload["data"].append(sample)
    
    # Make the request
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    elapsed = time.time() - start_time
    
    print(f"Status code: {response.status_code}")
    print(f"Response time: {elapsed:.4f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Processing time: {data['processing_time']:.4f} seconds")
        print(f"  Model version: {data['model_version']}")
        print(f"  Results count: {len(data['results'])}")
        
        # Display first result details
        if data['results']:
            first = data['results'][0]
            print(f"  Sample 0 threat level: {first['threat_level']}")
            print(f"  Sample 0 confidence: {first['confidence']:.4f}")
            print(f"  Sample 0 normal probability: {first['probabilities']['normal']:.4f}")
            print(f"  Sample 0 threat probability: {first['probabilities']['threat']:.4f}")
        return True
    return False

def test_training():
    """Test the training endpoint"""
    print("\n===== Testing Training Endpoint =====")
    
    # Create training request
    payload = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "dataset_size": 2000,
        "validation_split": 0.2
    }
    
    # Start training job
    response = requests.post(f"{BASE_URL}/train", json=payload)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        job_id = data['job_id']
        print(f"  Job ID: {job_id}")
        print(f"  Initial status: {data['status']}")
        
        # Poll job status a few times
        max_polls = 10
        for i in range(max_polls):
            print(f"\n  Poll {i+1}/{max_polls}: Checking status...")
            status_response = requests.get(f"{BASE_URL}/train/{job_id}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"  Status: {status_data['status']}")
                print(f"  Progress: {status_data['progress']:.2f}")
                print(f"  Message: {status_data['message']}")
                
                # If job completed or failed, show results and break
                if status_data['status'] in ['completed', 'failed']:
                    if status_data['results']:
                        print("\n  Training Results:")
                        for key, value in status_data['results'].items():
                            print(f"    {key}: {value}")
                    break
            else:
                print(f"  Error checking status: {status_response.status_code}")
                break
                
            # Wait before next poll
            time.sleep(0.5)
            
        return True
    return False

def test_batch_performance(batch_sizes=[1, 10, 50, 100]):
    """Test performance with different batch sizes"""
    print("\n===== Testing Batch Performance =====")
    
    results = []
    for size in batch_sizes:
        print(f"\nTesting batch size: {size}")
        
        # Create test payload
        payload = {"data": []}
        for i in range(size):
            sample = {f"feature_{j}": round(0.1 * ((i + j) % 10), 2) for j in range(39)}
            payload["data"].append(sample)
        
        # Make request and time it
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            processing_time = data["processing_time"]
            
            # Calculate metrics
            avg_per_sample = processing_time / size
            client_side_overhead = elapsed - processing_time
            
            print(f"  Status code: {response.status_code}")
            print(f"  Total response time: {elapsed:.4f} seconds")
            print(f"  Server processing time: {processing_time:.4f} seconds")
            print(f"  Average time per sample: {avg_per_sample:.6f} seconds")
            print(f"  Client-side overhead: {client_side_overhead:.4f} seconds")
            
            results.append({
                "batch_size": size,
                "total_time": elapsed,
                "processing_time": processing_time,
                "avg_per_sample": avg_per_sample,
                "overhead": client_side_overhead
            })
        else:
            print(f"  Failed with status code: {response.status_code}")
    
    # Print summary
    if results:
        print("\nPerformance Summary:")
        print("-------------------")
        print(f"{'Batch Size':^10} | {'Total Time':^12} | {'Process Time':^12} | {'Avg/Sample':^12} | {'Overhead':^12}")
        print("-" * 66)
        
        for r in results:
            print(f"{r['batch_size']:^10} | {r['total_time']:.6f} s | {r['processing_time']:.6f} s | {r['avg_per_sample']:.6f} s | {r['overhead']:.6f} s")
    
    return True

def main():
    print("===== NeuraShield AI API Testing =====")
    
    # Run the tests
    tests = [
        ("Health check", test_health),
        ("Metrics endpoint", test_metrics),
        ("Single sample analysis", lambda: test_analyze(1)),
        ("Batch analysis (10 samples)", lambda: test_analyze(10)),
        ("Training endpoint", test_training),
        ("Batch performance", test_batch_performance)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n\n{'=' * 50}")
        print(f"Running test: {name}")
        print(f"{'=' * 50}")
        
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"Error during test: {str(e)}")
            results.append((name, False))
    
    # Print summary
    print("\n\n")
    print("=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {name}")
    
    # Check if all tests passed
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\nAll tests passed successfully!")
        return 0
    else:
        print("\nSome tests failed. Check the output for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 