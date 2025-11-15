#!/usr/bin/env python3
"""
Evaluation script for the quantized LLM service.
Measures inference speed and compares output quality.
"""

import json
import time
import requests
from statistics import mean, stdev

API_URL = "http://localhost:8080"

# Test prompts for evaluation
TEST_PROMPTS = [
    "Artificial intelligence is",
    "The future of technology",
    "Machine learning enables",
    "Neural networks are used for",
    "Deep learning models can",
    "Natural language processing helps",
    "Computer vision allows",
    "Robotics and automation",
]

def test_generation(prompt: str, max_tokens: int = 30) -> dict:
    """Test a single generation and return metrics."""
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt, "max_new_tokens": max_tokens},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_speed_benchmark(num_runs: int = 5):
    """Run speed benchmark on all test prompts."""
    print("=" * 60)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 60)
    
    all_latencies = []
    all_throughputs = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: '{prompt}'")
        
        latencies = []
        throughputs = []
        
        # Warmup
        test_generation(prompt, max_tokens=20)
        
        # Run benchmark
        for run in range(num_runs):
            result = test_generation(prompt, max_tokens=20)
            if "error" not in result:
                latencies.append(result["total_time_ms"])
                throughputs.append(result["tokens_per_second"])
                print(f"  Run {run+1}: {result['total_time_ms']}ms, "
                      f"{result['tokens_per_second']:.1f} tokens/s")
        
        if latencies:
            avg_latency = mean(latencies)
            avg_throughput = mean(throughputs)
            all_latencies.extend(latencies)
            all_throughputs.extend(throughputs)
            
            print(f"  Average: {avg_latency:.1f}ms, {avg_throughput:.1f} tokens/s")
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total runs: {len(all_latencies)}")
    print(f"Average latency: {mean(all_latencies):.1f} ± {stdev(all_latencies):.1f} ms")
    print(f"Average throughput: {mean(all_throughputs):.1f} ± {stdev(all_throughputs):.1f} tokens/s")
    print(f"Min latency: {min(all_latencies):.1f} ms")
    print(f"Max latency: {max(all_latencies):.1f} ms")

def test_generation_quality():
    """Test generation quality with diverse prompts."""
    print("\n" + "=" * 60)
    print("GENERATION QUALITY TEST")
    print("=" * 60)
    
    for i, prompt in enumerate(TEST_PROMPTS[:4], 1):  # Test first 4
        print(f"\n[{i}] Prompt: '{prompt}'")
        result = test_generation(prompt, max_tokens=40)
        
        if "error" not in result:
            print(f"Generated: '{result['completion']}'")
            print(f"Tokens: {result['tokens_generated']}, "
                  f"Time: {result['total_time_ms']}ms")
        else:
            print(f"Error: {result['error']}")

def get_model_metadata():
    """Get and display model metadata."""
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/metadata")
        response.raise_for_status()
        data = response.json()
        
        if data.get("baseline"):
            baseline = data["baseline"]
            print(f"Model: {baseline['name']}")
            print(f"Type: {baseline['dtype']}")
            print(f"Quantized: {baseline['quantized']}")
            print(f"Size: {baseline['size_bytes'] / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error getting metadata: {e}")

if __name__ == "__main__":
    print("Starting LLM Service Evaluation")
    print("Server:", API_URL)
    
    # Check health
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.text.strip() == "ok":
            print("✓ Server is healthy\n")
        else:
            print("✗ Server returned unexpected response")
            exit(1)
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        exit(1)
    
    # Get model info
    get_model_metadata()
    
    # Run benchmarks
    run_speed_benchmark(num_runs=5)
    
    # Test quality
    test_generation_quality()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
