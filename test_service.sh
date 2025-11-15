#!/bin/bash
# Test script for the quantized LLM service

BASE_URL="http://localhost:8080"

echo "Testing Health Endpoint..."
curl -s "$BASE_URL/health"
echo -e "\n"

echo "Testing Metadata Endpoint..."
curl -s "$BASE_URL/metadata" | python3 -m json.tool
echo -e "\n"

echo "Testing Generate Endpoint..."
curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The quick brown fox", "max_new_tokens": 20}' | python3 -m json.tool
echo -e "\n"

echo "Testing Baseline Generate Endpoint..."
curl -s -X POST "$BASE_URL/generate/baseline" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Artificial intelligence is", "max_new_tokens": 25}' | python3 -m json.tool
